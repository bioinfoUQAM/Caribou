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
            'bacteria' : 1,
        }
        return self

    def _transform_pandas(self, df: pd.DataFrame):
        _validate_df(df, self.label_column)

        def column_label_encoder(s: pd.Series):
            s_values = self.stats_[f"unique_values({s.name})"]
            s = s.str.lower()
            s = s.map(s_values)
            s = s.fillna(-1)
            return s

        df[self.label_column] = df[self.label_column].transform(column_label_encoder)
        df = df.rename(columns = {self.label_column : LABELS_COLUMN_NAME})

        return df
