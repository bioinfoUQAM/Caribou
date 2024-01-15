
import numpy as np
import pandas as pd

from typing import List
from ray.data import Dataset
from math import ceil, floor
from ray.data.preprocessor import Preprocessor
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

TENSOR_COLUMN_NAME = '__value__'

class TensorOccurenceExclusion(Preprocessor):
    """
    Exclusion of the minimum & maximum occurences accross features to be used as a Ray preprocessor.
    """

    def __init__(self, features: List[str], num_features: int):
        # Parameters
        self.features = features
        self._nb_features = len(features)
        self._num_features = int(self._nb_features - num_features)

    def _fit(self, ds: Dataset) -> Preprocessor:
        # Nb of occurences
        occurences = np.zeros(self._nb_features)
        for batch in ds.iter_batches(batch_format = 'numpy'):
            batch = batch[TENSOR_COLUMN_NAME]
            occurences += np.count_nonzero(batch, axis = 0)

        # Include / Exclude by sorted position
        cols_keep = pd.Series(occurences, index = self.features)
        cols_keep = cols_keep.sort_values(ascending = True) # Long operation
        cols_keep = cols_keep.iloc[0 : self._num_features]
        cols_keep = list(cols_keep.index)

        # self.stats_ = {'cols_keep' : cols_keep, 'cols_drop' : cols_drop}
        self.stats_ = {'cols_keep' : cols_keep}

        return self

    def _transform_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        # _validate_df(df, TENSOR_COLUMN_NAME, self._nb_features)
        cols_keep = self.stats_['cols_keep']
        
        tensor_col = df[TENSOR_COLUMN_NAME]
        tensor_col = _unwrap_ndarray_object_type_if_needed(tensor_col)
        tensor_col = pd.DataFrame(tensor_col, columns = self.features)

        tensor_col = tensor_col[cols_keep].to_numpy()
        
        df[TENSOR_COLUMN_NAME] = pd.Series(list(tensor_col))

        return df
        
    def __repr__(self):
        return (f"{self.__class__.__name__}(features={self._nb_features!r}, num_features={self._num_features!r})")

class TensorPercentOccurenceExclusion(Preprocessor):
    """
    Exclusion of the features present in less than (%) / more than (100% - %) across samples to be used as a Ray preprocessor.
    """

    def __init__(self, features: List[str], percent : int = 0.05):
        # Parameters
        self.features = features
        self.percent = percent
        self._nb_features = len(features)
    
    def _fit(self, ds: Dataset) -> Preprocessor:
        nb_samples = ds.count()
        high_treshold = floor((1 -  self.percent) * nb_samples)
        occurences = np.zeros(self._nb_features)

        # Function for parallel occurences counting
        def count_occurences(batch):
            batch = batch[TENSOR_COLUMN_NAME]
            batch = _unwrap_ndarray_object_type_if_needed(batch)
            return {'occurences' : [np.count_nonzero(batch, axis = 0)]}
        
        occur = ds.map_batches(count_occurences, batch_format = 'numpy')

        for row in occur.iter_rows():
            occurences += row['occurences']

        # Construct list of features to keep by position
        cols_keep = [self.features[i] for i, occurence in enumerate(occurences) if occurence < high_treshold]
        
        if 0 < len(cols_keep) :
            self.stats_ = {'cols_keep' : cols_keep}
        else:
            self.stats_ = {'cols_keep' : self.features}

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
        return (f"{self.__class__.__name__}(features={self._nb_features!r}, percent={self.percent!r}%)")

def _validate_df(df: pd.DataFrame, column: str, nb_features: int) -> None:
    if len(df.loc[0, column]) != nb_features:
        raise ValueError('Discordant number of features in the tensor column with the one from the dataframe used for fitting')