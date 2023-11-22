
import numpy as np
import pandas as pd

from typing import List
from warnings import warn
from os.path import isfile
from ray.data import Dataset
from utils import save_Xy_data, load_Xy_data

from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import DictionaryLearning, NMF, MiniBatchNMF

from ray.data.preprocessor import Preprocessor
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

TENSOR_COLUMN_NAME = '__value__'

class TensorNMFDecomposition(Preprocessor):
    """
    Custom class for using Mini-Batch Non-Negative Matrix Factorization (NMF) as a Ray preprocessor.
    This is inspired by sklearn.decomposition.NMF and is fitted on batches before keeping the consensus components matrix.
    Consensus components matrix is attained following the logic from sklearn.decomposition.MiniBatchNMF.
    https://scikit-learn.org/stable/modules/decomposition.html#nmf
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchNMF.html
    """
    def __init__(self, features: List[str], nb_components: int = 10000, file: str = ''):
        # Parameters
        self.features = features
        self._nb_features = len(features)
        self._nb_components = nb_components
        self._file = file

    def _fit(self, ds: Dataset) -> Preprocessor:
        def batch_nmf(batch):
            batch = batch[TENSOR_COLUMN_NAME]
            batch = _unwrap_ndarray_object_type_if_needed(batch)
            model = NMF(
                n_components = self._nb_components,
                init = 'random'
            )
            model.fit(batch)
            return {'components' : [model.components_]}

        components = []
        if self._nb_features > self._nb_components:
            if isfile(self._file):
                components = np.array(load_Xy_data(self._file))
            else:
                nmf = ds.map_batches(batch_nmf, batch_format = 'numpy')
                
                for row in nmf.iter_rows():
                    components.append(row['components'])
                components = np.mean(components, axis = 0)

                save_Xy_data(components, self._file)

            self.stats_ = {'components' : components}
        else:
            warn('No features reduction to do because the number of features is already lower than the required number of components')
            self.stats_ = {'components' : False}
    
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
        return (f"{self.__class__.__name__}(features={self._nb_features!r}, file={self._file!r})")

def _validate_df(df: pd.DataFrame, column: str, nb_features: int) -> None:
    if len(df.loc[0, column]) != nb_features:
        raise ValueError('Discordant number of features in the tensor column with the one from the dataframe used for fitting')
