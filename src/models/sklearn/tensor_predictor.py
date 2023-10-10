from typing import List, Optional, Union

import pandas as pd
from joblib import parallel_backend
from sklearn.base import BaseEstimator

from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed
from ray.train.sklearn._sklearn_utils import _set_cpu_params
from ray.util.joblib import register_ray

from ray.data.preprocessor import Preprocessor
from ray.train.sklearn import SklearnPredictor

TENSOR_COLUMN_NAME = '__value__'

class SklearnTensorPredictor(SklearnPredictor):
    """
    Class adapted from Ray's SklearnPredictor class to allow for use with tensor column.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        preprocessor: Optional[Preprocessor] = None,
    ):
        super().__init__(estimator, preprocessor)

    def _predict_pandas(
        self,
        data: pd.DataFrame,
        features: List[str],
        feature_columns: Optional[Union[List[str], List[int]]] = None,
        num_estimator_cpus: Optional[int] = 1,
        **predict_kwargs,
    ) -> pd.DataFrame:
        register_ray()

        if num_estimator_cpus:
            _set_cpu_params(self.estimator, num_estimator_cpus)

        data = data[TENSOR_COLUMN_NAME]
        data = _unwrap_ndarray_object_type_if_needed(data)
        data = pd.DataFrame(data, columns = features)

        with parallel_backend("ray", n_jobs=num_estimator_cpus):
            df = pd.DataFrame(self.estimator.predict(data, **predict_kwargs))
        df.columns = (
            ["predictions"]
            if len(df.columns) == 1
            else [f"predictions_{i}" for i in range(len(df.columns))]
        )
        return df