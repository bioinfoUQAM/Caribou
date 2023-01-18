
import numpy as np
import pandas as pd

from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors.encoder import _get_unique_value_indices, _validate_df

class OneHotTensor(Preprocessor):
    """
    Custom implementation of Ray's OneHotEncoder to encode directly into a tensor in ray.data.dataset.Dataset.
    """
    def __init__(
        self,
        column: str,
    ):
        self.columns = column

    def _fit(self, dataset: Dataset) -> Preprocessor:
        self.stats_ = _get_unique_value_indices(
            dataset,
            self.columns,
            encode_lists = False,
        )

        return self

    def _transform_pandas(self, df: pd.DataFrame):
        _validate_df(df)

        values = self.stats_[f"unique_values({self.column})"]
        unique = list(np.unique(values))

        df['labels'] = [np.zeros(len(unique)) for i in range(len(df))]
        


        return df

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(columns={self.columns!r}, "
            f"max_categories={self.max_categories!r})"
        )

