
import numpy as np
import pandas as pd

from ray.data.dataset import Dataset
from ray.data.preprocessor import Preprocessor
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

TENSOR_COLUMN_NAME = '__value__'

class TensorMinMaxScaler(Preprocessor):
    """
    Custom implementation of Ray's MinMax Scaler for usage with tensor column in ray.data.dataset.Dataset.
    """
    
    def __init__(self, nb_features):
        # Parameters
        self.__nb_features = nb_features
        
    def _fit(self, ds: Dataset) -> Preprocessor:
        """
        Fit the MinMaxScaler to the given dataset.
        """
        min = []
        max = []

        def Min(dct):
            arr = dct[TENSOR_COLUMN_NAME]
            min = np.array([arr[:,i].min() for i in range(self.__nb_features)])
            return min

        def Max(dct):
            arr = dct[TENSOR_COLUMN_NAME]
            max = np.array([arr[:,i].max() for i in range(self.__nb_features)])
            return max

        for batch in ds.iter_batches(batch_format = 'numpy'):
            min.append(Min(batch))
            max.append(Max(batch))

        min = np.array(min)
        max = np.array(max)

        min = np.array([min[:,i].min() for i in range(self.__nb_features)])
        max = np.array([max[:,i].max() for i in range(self.__nb_features)])
                
        self.stats_ = {'min' : min, 'max' : max}

        return self

    def _transform_pandas(self, batch: pd.DataFrame):
        """
        Transform the given dataset to pandas dataframe.
        """
        min = self.stats_['min']
        max = self.stats_['max']
        df = batch[TENSOR_COLUMN_NAME]
        df = _unwrap_ndarray_object_type_if_needed(df)

        diff = max - min
        diff[diff == 0] = 1

        batch[TENSOR_COLUMN_NAME] = pd.Series(list((df - min) / diff))

        return batch

    def _transform_numpy(self, batch: dict):
        """
        Transform the given dataset to numpy ndarray.
        """

        min = self.stats_['min']
        max = self.stats_['max']

        diff = max - min

        batch[TENSOR_COLUMN_NAME] = (batch[TENSOR_COLUMN_NAME] - min) / diff

        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}(columns={self._nb_features!r})"
