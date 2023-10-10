
import numpy as np
import pandas as pd

from typing import List
from warnings import warn
from ray.data import Dataset
from sklearn.feature_selection import chi2
from ray.data.preprocessor import Preprocessor
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

TENSOR_COLUMN_NAME = '__value__'

class TensorChi2Selection(Preprocessor):
    """
    Custom implementation of SelectKBest with Chi2 inspired by sklearn.feature_selection.SelectPercentile and sklearn.feature_selection.chi2 features selector to be used as a Ray preprocessor.
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest
    """

    def __init__(self, features: List[str], threshold: float = 0.05):
        # Parameters
        self.features = features
        self.threshold = threshold
        self._nb_features = len(features)

    def _fit(self, ds: Dataset) -> Preprocessor:
        mean_chi = []
        cols_keep = []
        cols_drop = []
        # Compute chi2 over batches
        for batch in ds.iter_batches(batch_size = 5, batch_format = 'pandas'):
            X = batch[TENSOR_COLUMN_NAME].to_numpy()
            X = _unwrap_ndarray_object_type_if_needed(X)
            X = pd.DataFrame(X, columns = self.features)
            y = batch['species'].to_numpy().ravel()
            mean_chi.append(chi2(X, y)[1])

        # Compute the mean of chi2 by feature
        mean_chi = np.array(mean_chi)
        mean_chi = np.mean(mean_chi, axis = 0)

        cols_keep = pd.Series(mean_chi, index = self.features)
        cols_keep = cols_keep[cols_keep <= self.threshold]
        cols_keep = list(cols_keep.index)
        
        # Keep all features if none are under the threshold
        if len(cols_keep) == 0:
            cols_keep = self.features
            warn('No values were found to have a chi2 p-value under the threshold, all features will be kept.\
                 You can try running this feature selector again with a different threshold to reduce the number of features')
        else:
            cols_drop = list(set(self.features).difference(set(cols_keep)))

        self.stats_ = {'cols_keep' : cols_keep, 'cols_drop' : cols_drop}
        return self
    
    def _transform_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        _validate_df(df, TENSOR_COLUMN_NAME, self._nb_features)
        cols_drop = self.stats_['cols_drop']
        
        tensor_col = df[TENSOR_COLUMN_NAME]
        tensor_col = _unwrap_ndarray_object_type_if_needed(tensor_col)
        tensor_col = pd.DataFrame(tensor_col, columns = self.features)

        tensor_col = tensor_col.drop(cols_drop, axis = 1)
        tensor_col = tensor_col.to_numpy()

        df[TENSOR_COLUMN_NAME] = pd.Series(list(tensor_col))

        return df
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(features={self._nb_features!r}, threshold={self.threshold!r})")

def _validate_df(df: pd.DataFrame, column: str, nb_features: int) -> None:
    if len(df.loc[0, column]) != nb_features:
        raise ValueError('Discordant number of features in the tensor column with the one from the dataframe used for fitting')