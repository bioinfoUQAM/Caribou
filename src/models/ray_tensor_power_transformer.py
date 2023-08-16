
import ray
import numpy as np
import pandas as pd

from typing import List
from ray.data.preprocessor import Preprocessor
from sklearn.preprocessing import PowerTransformer
from ray.data.extensions.tensor_extension import TensorArray

class TensorPowerTransformer(Preprocessor):
    """
    Custom implementation of Ray's PowerTransformer for usage with Numpy tensors column in ray.data.dataset.Dataset.
    """
    def __init__(self, features_list: List[str]):
        self._features_list = features_list
        self.method = "yeo-johnson"
        self.stats_ = {}

    def _fit(self, ds: ray.data.dataset.Dataset):
        """
        Fit the MinMaxScaler to the given dataset.
        """
        nb_samples = ds.count()
        dct_values = {}
        for feature in self._features_list:
            dct_values[feature] = np.zeros(nb_samples, dtype = np.int32)
        
        previous_pos = 0
        # Get values per column
        for batch in ds.iter_batches(batch_format = 'numpy'):
            batch = batch['__value__']
            batch_size = len(batch)
            for i, feature in enumerate(self._features_list):
                dct_values[feature][previous_pos:(previous_pos+batch_size)] = batch[:,i]
            previous_pos = previous_pos + batch_size
        
        # Fit the scalers on each column
        for feature, tensor in dct_values.items():
            scaler = PowerTransformer()
            scaler.fit(tensor.reshape(-1,1))
            self.stats_[feature] = scaler

        # def extract_col_val(batch : np.array, col : str):
        #     batch = pd.DataFrame(batch, columns = self._features_list)
        #     return batch[col].values

        # for feature in self._features_list:
        #     ds_col = ds.map_batches(
        #         lambda batch: extract_col_val(batch['__value__'], feature),
        #         batch_format = 'numpy'
        #     )
        #     ds_col = ds_col.to_pandas()
        #     ds_col = ds_col.to_numpy().reshape(-1,1)
        #     transformer = PowerTransformer(
        #         method = self.method
        #     )
        #     transformer.fit(ds_col)
        #     self.stats_[feature] = transformer
        #     self.fitted_ = True
        
        return self
        
    def _transform_pandas(self, batch: pd.DataFrame):
        """
        Transform the given dataset to pandas dataframe.
        """
        df = pd.DataFrame(np.vstack(batch['__value__']), columns = self._features_list)
        for feature, transformer in self.stats_.items():
            transformed = df[feature].to_numpy().reshape(-1,1)
            transformed = transformer.transform(transformed)
            df[feature] = transformed
    
        batch['__value__'] = TensorArray(df.to_numpy(dtype = np.float32))

        return batch
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(columns={self.columns!r}, "
            f"PowerTransformersCollection={self.stats_!r}, method={self.method!r})"
        )

