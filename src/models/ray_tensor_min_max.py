from typing import List, Tuple

import ray
import numpy as np
import pandas as pd

from ray.data.preprocessor import Preprocessor
from ray.data.extensions.tensor_extension import TensorArray

class TensorMinMaxScaler(Preprocessor):
    """
    Custom implementation of Ray's MinMax Scaler for usage with Numpy tensors column in ray.data.dataset.Dataset.
    """
    
    def __init__(self, features_list):
        # Parameters
        self._features_list = features_list
        # Empty inits
        self._min = None
        self._max = None

    def _fit(self, dataset:ray.data.dataset.Dataset):
        """
        Fit the MinMaxScaler to the given dataset.
        """
        self._min = np.full((len(self._features_list)), np.inf)
        self._max = np.zeros(len(self._features_list), dtype = np.int32)
        for batch in dataset.iter_batches(batch_format = 'numpy'):
            for i in np.arange(len(self._features_list)):
                local_min = min(batch['__value__'][:,i])
                local_max = max(batch['__value__'][:,i])
                if local_min < self._min[i]:
                    self._min[i] = local_min
                if local_max > self._max[i]:
                    self._max[i] = local_max
        
        self.fitted_ = True

        return self

    def _transform_pandas(self, batch: pd.DataFrame):
        """
        Transform the given dataset to pandas dataframe.
        """
        df = pd.DataFrame(np.vstack(batch['__value__']), columns = self._features_list)

        for i, col in enumerate(self._features_list):
            df[col] = df[col].apply(value_transform, args=(self._min[i], self._max[i]))

        batch['__value__'] = TensorArray(np.array(df))

        return batch

    def _transform_numpy(self, batch: dict):
        """
        Transform the given dataset to numpy ndarray.
        """
        df = np.array(batch['__value__'], dtype = np.float32)
        vecfunc = np.vectorize(value_transform)
        for i in np.arange(len(self._features_list)):
            df[:,i] = vecfunc(df[:,i], self._min[i], self._max[i])

        batch['__value__'] = df

        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}(columns={self._features_list!r})"

# Function to map to the data, used by both data representations
def value_transform(x, _min, _max):
    return (x - _min) / (_max - _min)