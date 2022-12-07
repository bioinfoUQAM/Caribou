from collections import OrderedDict
import pandas as pd

from ray.data.dataset import Dataset
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors.encoder import _validate_df

class OneClassSVMLabelEncoder(Preprocessor):
    """
    Encode labels as integer targets for Scikit-Learn SGDOneClassSVM model.

    """

    def __init__(self, label_column: str):
        self.label_column = label_column

    def _fit(self, dataset : Dataset) -> Preprocessor:
        mapping = OrderedDict()
        mapping[f"unique_values({self.label_column})"] = {
            'bacteria': 1,
            'unknown': -1
        }
        self.stats_ = mapping
        return self

    def _transform_pandas(self, df: pd.DataFrame):
        _validate_df(df, self.label_column)

        def column_label_encoder(s: pd.Series):
            s_values = self.stats_[f"unique_values({s.name})"]
            return s.map(s_values)

        df[self.label_column] = df[self.label_column].transform(column_label_encoder)
        return df

    def __repr__(self):
        return f"{self.__class__.__name__}(label_column={self.label_column!r})"
