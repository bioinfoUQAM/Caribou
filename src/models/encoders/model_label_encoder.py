from collections import Counter, OrderedDict
from functools import partial
from typing import Dict, List, Optional

import ray
import numpy as np
import pandas as pd
import pandas.api.types

from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors.encoder import _get_unique_value_indices

LABELS_COLUMN_NAME = 'labels'

class ModelLabelEncoder(Preprocessor):
    """
    Custom implementation of Ray's LabelEncoder to set column name as it encodes labels.
    """
    def __init__(self, label_column: str):
        self.label_column = label_column

    def _fit(self, dataset: Dataset) -> Preprocessor:
        self.stats_ = _get_unique_value_indices(dataset, [self.label_column])
        return self

    def _transform_pandas(self, df: pd.DataFrame):
        def column_label_encoder(s: pd.Series):
            s_values = self.stats_[f"unique_values({s.name})"]
            return s.map(s_values)

        df[self.label_column] = df[self.label_column].transform(column_label_encoder)
        df = df.rename(columns = {self.label_column : LABELS_COLUMN_NAME})

        return df

    def __repr__(self):
        return f"{self.__class__.__name__}(label_column={self.label_column!r})"