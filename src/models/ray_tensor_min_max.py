from typing import List, Tuple

import ray
import numpy as np
import pandas as pd

from ray.data.dataset import Dataset
from ray.data.preprocessor import Preprocessor
from ray.data.extensions.tensor_extension import TensorArray

class TensorMinMaxScaler(Preprocessor):
    """
    Custom implementation of Ray's MinMax Scaler for usage with Numpy tensors column in ray.data.dataset.Dataset.
    """
    
    def __init__(self, features_list):
        # Parameters
        self._features_list = features_list
        
    def _fit(self, ds: Dataset) -> Preprocessor:
        """
        Fit the MinMaxScaler to the given dataset.
        """
        min = []
        max = []
        nb_features = len(self._features_list)

        def Min(dct):
            arr = dct['__value__']
            min = np.array([arr[:,i].min() for i in range(nb_features)])
            return min

        def Max(dct):
            arr = dct['__value__']
            max = np.array([arr[:,i].max() for i in range(nb_features)])
            return max

        for batch in ds.iter_batches(batch_format = 'numpy'):
            min.append(Min(batch))
            max.append(Max(batch))

        min = np.array(min)
        max = np.array(max)

        min = np.array([min[:,i].min() for i in range(nb_features)])
        max = np.array([max[:,i].max() for i in range(nb_features)])
                
        self.stats_ = {'min' : min, 'max' : max}

        return self

    def _transform_pandas(self, batch: pd.DataFrame):
        """
        Transform the given dataset to pandas dataframe.
        """
        min = self.stats_['min']
        max = self.stats_['max']
        df = np.vstack(batch['__value__'].to_numpy())

        diff = max - min
        diff[diff == 0] = 1

        batch['__value__'] = pd.Series(list((df - min) / diff))

        return batch

    def _transform_numpy(self, batch: dict):
        """
        Transform the given dataset to numpy ndarray.
        """

        min = self.stats_['min']
        max = self.stats_['max']

        diff = max - min

        batch['__value__'] = (batch['__value__'] - min) / diff

        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}(columns={self._features_list!r})"

# Function to map to the data, used by both data representations
def value_transform(x, _min, _max):
    return (x - _min) / (_max - _min)