
import numpy as np
import pandas as pd

from ray.data.dataset import Dataset
from ray.data.preprocessor import Preprocessor

TENSOR_COLUMN_NAME = '__value__'

class ComputeClassWeights(Preprocessor):
    """
    Custom implementation of Class Weight Computation inspired by sklearn.utils.class_weight.compute_class_weight to be used as a Ray preprocessor.
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    This permits to estimate balanced class weights for an unbalanced dataset.
    """

    def __init__(self, class_col):
        # Parameters
        self._col = class_col
        self._cls = []
        self._counts_map = {}

    def _fit(self, ds: Dataset) -> Preprocessor:
        def get_cls_counts(df):
            mapping = {}
            counts = df[self._col].value_counts()
            for cls in self._cls:
                if cls in counts.index:
                    mapping[str(cls)] = [counts[cls]]
                else:
                    mapping[str(cls)] = [0]
            return mapping
        
        self._cls = ds.unique(self._col)
        
        counts = ds.map_batches(get_cls_counts, batch_format = 'pandas')
                
        for cls in self._cls:
            self._counts_map[str(cls)] = counts.sum(str(cls))

        freqs = ds.count() / (len(self._cls) * np.array(list(self._counts_map.values())).astype(np.float64))
        
        self.stats_ = {}
        for i, cls in enumerate(self._cls):
            self.stats_[cls] = freqs[i]
                
        return self

        
