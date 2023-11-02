import numpy as np
import pandas as pd

from typing import List
from warnings import warn
from ray.data import Dataset

from sklearn.utils.extmath import randomized_svd

from ray.data.preprocessor import Preprocessor
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

TENSOR_COLUMN_NAME = '__value__'

class TensorTruncatedSVDReduction(Preprocessor):
    """
    Custom class for using a mix of TruncatedSVD inspired by sklearn.decomposition.TruncatedSVD and applying a batched strategy inspired by sklearn.decomposition.IncrementalPCA to process batches in parallel.
    This makes it possible to use the class as a Ray preprocessor in a features reduction strategy.
    TruncatedSVD performs linear dimensionality reduction by means of truncated singular value decomposition (SVD).
    When it is applied following the TF-IDF normalisation, it becomes a latent semantic analysis (LSA).
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD
    """
    
    def __init__(self, features: List[str], nb_components: int = 10000):
        # Parameters
        self.features = features
        self._nb_features = len(features)
        self._nb_components = nb_components
        
    
    def _fit(self, ds: Dataset) -> Preprocessor:
        def svd_batch(arr: np.array):
            df = arr['__value__']
            df = _unwrap_ndarray_object_type_if_needed(df)
            U, Sigma, VT = randomized_svd(
                df,
                n_components = self._nb_components,
                n_iter = 5,
                n_oversamples = 10,
                power_iteration_normalizer = 'LU',
                random_state = None
            )

            return {'VT': [VT]}

        if self._nb_features > self._nb_components:
            # Exec svd
            components = []
            svd_vt = ds.map_batches(svd_batch, batch_format = 'numpy')

            for row in svd_vt.iter_rows():
                components.append(row['VT'])

            components = np.mean(components, axis = 0)

            self.stats_ = {'components' : components}
        else:
            warn('No features reduction to do because the number of features is already lower than the required number of components')
            self.stats_ = {'components' : False}

        return self

    def _transform_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        # _validate_df(df, TENSOR_COLUMN_NAME, self._nb_features)
        components = self.stats_['components']
        
        if components is not False:
            tensor_col = df[TENSOR_COLUMN_NAME]
            tensor_col = _unwrap_ndarray_object_type_if_needed(tensor_col)
            tensor_col = np.dot(tensor_col, components.T)
            df[TENSOR_COLUMN_NAME] = pd.Series(list(tensor_col))        

        return df

    def __repr__(self):
        return (f"{self.__class__.__name__}(features={self._nb_features!r}, taxa={self.taxa!r}, threshold={self.threshold!r})")

def _validate_df(df: pd.DataFrame, column: str, nb_features: int) -> None:
    if len(df.loc[0, column]) != nb_features:
        raise ValueError('Discordant number of features in the tensor column with the one from the dataframe used for fitting')
