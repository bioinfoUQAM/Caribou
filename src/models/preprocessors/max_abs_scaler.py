
import ray
import numpy as np
import pandas as pd

from ray.data.preprocessor import Preprocessor
from ray.data.extensions.tensor_extension import TensorArray
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

TENSOR_COLUMN_NAME = '__value__'

class TensorMaxAbsScaler(Preprocessor):
    """
    Custom implementation of Ray's MaxAbsScaler for usage with tensor column in ray.data.dataset.Dataset.
    """
    
    def __init__(self, features):
        # Parameters
        self._features = features
        # Empty inits
        self._absmax = None

    def _fit(self, dataset:ray.data.dataset.Dataset):
        """
        Fit the MaxAbsScaler to the given dataset.
        """
        self._absmax = np.zeros(len(self._features), dtype = np.int32)
        for batch in dataset.iter_batches(batch_format = "numpy"):
            for i in np.arange(len(self._features)):
                local_max = max(batch[TENSOR_COLUMN_NAME][:,i])
                if local_max > self._absmax[i]:
                    self._absmax[i] = local_max
        
        self.fitted_ = True

        return self

    def _transform_pandas(self, batch: pd.DataFrame):
        """
        Transform the given dataset to pandas dataframe.
        """
        df = batch[TENSOR_COLUMN_NAME]
        df = _unwrap_ndarray_object_type_if_needed(df)
        df = pd.DataFrame(df, columns = self._features)
        for i, col in enumerate(self._features):
            df[col] = df[col].apply(value_transform, args=[self._absmax[i]])

        batch[TENSOR_COLUMN_NAME] = TensorArray(np.array(df))

        return batch

    def _transform_numpy(self, batch: dict):
        """
        Transform the given dataset to numpy ndarray.
        """
        df = np.array(batch[TENSOR_COLUMN_NAME], dtype = np.float32)
        vecfunc = np.vectorize(value_transform)
        for i in np.arange(len(self._features)):
            df[:,i] = vecfunc(df[:,i], self._absmax[i])

        batch[TENSOR_COLUMN_NAME] = df

        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}(columns={self._features!r})"

# Function to map to the data, used by both data representations
def value_transform(x, _min, _max):
    return (x - _min) / (_max - _min)