import numpy as np
import pandas as pd

import collections

from typing import List
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors.utils import simple_hash
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

TENSOR_COLUMN_NAME = '__value__'

class TensorCountHashing(Preprocessor):
    """
    Class adapted from ray.data.preprocessors.FeatureHasher to use with tensors
    https://docs.ray.io/en/releases-2.6.3/_modules/ray/data/preprocessors/hasher.html#FeatureHasher
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html#sklearn.feature_extraction.FeatureHasher
    """
    _is_fittable = False

    def __init__(self, features: List[str], num_features: int):
        self.features = features
        self.num_features = num_features

    def _transform_pandas(self, df: pd.DataFrame):
        def row_feature_hasher(row):
            hash_counts = collections.defaultdict(int)
            for feature in self.features:
                hashed_value = simple_hash(feature, self.num_features)
                hash_counts[hashed_value] += row[feature]
            return {f"hash_{i}": hash_counts[i] for i in range(self.num_features)}

        tensor_col = df[TENSOR_COLUMN_NAME]
        tensor_col = _unwrap_ndarray_object_type_if_needed(tensor_col)
        tensor_col = pd.DataFrame(tensor_col, columns = self.features)

        tensor_col = tensor_col.apply(
            row_feature_hasher, axis=1, result_type="expand"
        )
        
        tensor_col = tensor_col.to_numpy()

        df[TENSOR_COLUMN_NAME] = pd.Series(list(tensor_col))

        return df

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(columns={self.columns!r}, "
            f"num_features={self.num_features!r})"
        )