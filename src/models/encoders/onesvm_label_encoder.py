import numpy as np
import pandas as pd

from collections import OrderedDict
from ray.data.dataset import Dataset
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors.encoder import _get_unique_value_indices, _validate_df, LabelEncoder

LABELS_COLUMN_NAME = 'labels'

class OneClassSVMLabelEncoder(LabelEncoder):
    """
    Class adapted from Ray's LabelEncoder class to encode labels as integer targets for Scikit-Learn SGDOneClassSVM model.

    """

    def __init__(self, label_column: str):
        self.label_column = label_column

    def _fit(self, dataset : Dataset) -> Preprocessor:
        self.stats_ = OrderedDict()
        self.stats_[f"unique_values({self.label_column})"] = {
            'Bacteria' : 1
        }
        return self

    def _transform_pandas(self, df: pd.DataFrame):
        _validate_df(df, self.label_column)
        mapping = self.stats_[f"unique_values({self.label_column})"]
        df[self.label_column] = df[self.label_column]
        df[self.label_column] = df[self.label_column].map(mapping)
        df[self.label_column] = df[self.label_column].fillna(-1)

        df = df.rename(columns = {self.label_column : LABELS_COLUMN_NAME})

        return df
