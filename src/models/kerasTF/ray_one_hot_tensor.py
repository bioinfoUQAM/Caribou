
import numpy as np
import pandas as pd

from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
from ray.data.extensions.tensor_extension import TensorArray
from ray.data.preprocessors.encoder import _get_unique_value_indices, _validate_df

class OneHotTensorEncoder(Preprocessor):
    """
    Custom implementation of Ray's OneHotEncoder to encode directly into a tensor in ray.data.dataset.Dataset.
    """
    def __init__(
        self,
        column: str,
    ):
        self.column = column

    def _fit(self, dataset: Dataset) -> Preprocessor:
        self.stats_ = _get_unique_value_indices(
            dataset,
            [self.column],
            encode_lists = False,
        )

        return self

    def _transform_pandas(self, df: pd.DataFrame):
        df = _validate_df(df, self.column)

        def tensor_col_encoding(label, nb_unique):
            if label == -1:
                return TensorArray(np.zeros(nb_unique, dtype = np.int32))
            else:
                tensor = np.zeros(nb_unique, dtype = np.int32)
                tensor[label] = 1
                return TensorArray(tensor)

        values = self.stats_[f"unique_values({self.column})"]
        unique = list(values.keys())

        df = df.assign(labels=lambda x: [tensor_col_encoding(x.loc[ind,self.column], len(unique)) for ind in df.index])

        return df

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(columns={self.column!r}, "
            f"max_categories={self.max_categories!r})"
        )


def _validate_df(df: pd.DataFrame, column: str) -> None:
    if df[column].isna().values.any():
        df = df.fillna(-1)
    return df
