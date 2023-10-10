from typing import List

import ray
import numpy as np
import pandas as pd

from ray.data.preprocessor import Preprocessor
from sklearn.preprocessing import PowerTransformer
from ray.data.extensions.tensor_extension import TensorArray

TENSOR_COLUMN_NAME = '__value__'

class TensorPowerTransformer(Preprocessor):
    """
    Custom implementation of Ray's PowerTransformer for usage with tensor column in ray.data.dataset.Dataset.
    """
    def __init__(self, features_list: List[str]):
        self._features_list = features_list
        self.method = "yeo-johnson"
        self.stats_ = {}

    def _fit(self, ds: ray.data.dataset.Dataset):
        """
        Fit the PowerTransformer to the given dataset.
        """
        nb_samples = ds.count()
        dct_values = {}
        for feature in self._features_list:
            dct_values[feature] = np.zeros(nb_samples, dtype = np.int32)
        
        previous_pos = 0
        # Get values per column
        for batch in ds.iter_batches(batch_format = 'numpy'):
            batch = batch[TENSOR_COLUMN_NAME]
            batch_size = len(batch)
            for i, feature in enumerate(self._features_list):
                dct_values[feature][previous_pos:(previous_pos+batch_size)] = batch[:,i]
            previous_pos = previous_pos + batch_size
        
        # Fit the scalers on each column
        for feature, tensor in dct_values.items():
            scaler = PowerTransformer()
            scaler.fit(tensor.reshape(-1,1))
            self.stats_[feature] = scaler
        
        return self
        
    def _transform_pandas(self, batch: pd.DataFrame):
        """
        Transform the given dataset to pandas dataframe.
        """
        df = pd.DataFrame(np.vstack(batch[TENSOR_COLUMN_NAME]), columns = self._features_list)
        for feature, transformer in self.stats_.items():
            transformed = df[feature].to_numpy().reshape(-1,1)
            transformed = transformer.transform(transformed)
            df[feature] = transformed
    
        batch[TENSOR_COLUMN_NAME] = TensorArray(df.to_numpy(dtype = np.float32))

        return batch
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(columns={self.columns!r}, "
            f"PowerTransformersCollection={self.stats_!r}, method={self.method!r})"
        )