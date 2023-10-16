import logging

import numpy as np
import pandas as pd

from typing import List
from warnings import warn
from ray.data import Dataset

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif, f_oneway

from ray.data.preprocessor import Preprocessor
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

TENSOR_COLUMN_NAME = '__value__'

class TensorFeaturesSelection(Preprocessor):
    """
    Custom implementation of SelectKBest with Chi2 inspired by sklearn.feature_selection.SelectPercentile and sklearn.feature_selection.chi2 features selector to be used as a Ray preprocessor.
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest
    """

    def __init__(self, features: List[str], taxa: str, threshold: float = 0.5):
        # Parameters
        self.taxa = taxa
        self.features = features
        self.threshold = threshold
        self._nb_features = len(features)

    def _fit(self, ds: Dataset) -> Preprocessor:
        # Function for parallel stats computing
        def stats(batch):
            X = batch[TENSOR_COLUMN_NAME]
            X = _unwrap_ndarray_object_type_if_needed(X)
            X = pd.DataFrame(X, columns = self.features)
            y = batch[self.taxa].ravel()
            return {'chi' : [chi2(X, y)[0]]}

        mean_chi = []
        cols_keep = []
        
        # Chi batches means extraction
        chi = ds.map_batches(stats, batch_format = 'numpy', batch_size = 32)
        for i, row in enumerate(chi.iter_rows()):
            mean_chi.append(row['chi'])

        # Chi mean of batches means computing
        mean_chi = np.array(mean_chi)
        mean_chi = np.nanmean(mean_chi, axis = 0)

        # Determine the threshold from distribution of chi values
        self.threshold = np.nanquantile(mean_chi, self.threshold)
        
        # Keep features with values higher than the threshold
        cols_keep = [self.features[i] for i, chi in enumerate(mean_chi) if chi > self.threshold]
        
        self.stats_ = {'cols_keep' : cols_keep}

        return self

    def _transform_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        # _validate_df(df, TENSOR_COLUMN_NAME, self._nb_features)
        cols_keep = self.stats_['cols_keep']

        if len(cols_keep) < self._nb_features:
            tensor_col = df[TENSOR_COLUMN_NAME]
            tensor_col = _unwrap_ndarray_object_type_if_needed(tensor_col)
            tensor_col = pd.DataFrame(tensor_col, columns = self.features)

            tensor_col = tensor_col[cols_keep].to_numpy()

            df[TENSOR_COLUMN_NAME] = pd.Series(list(tensor_col))        

        return df

    def __repr__(self):
        return (f"{self.__class__.__name__}(features={self._nb_features!r}, taxa={self.taxa!r}, threshold={self.threshold!r})")

def _validate_df(df: pd.DataFrame, column: str, nb_features: int) -> None:
    if len(df.loc[0, column]) != nb_features:
        raise ValueError('Discordant number of features in the tensor column with the one from the dataframe used for fitting')