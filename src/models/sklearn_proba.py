import numpy as np
import pandas as pd
from joblib import parallel_backend

from ray.air.constants import TENSOR_COLUMN_NAME
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed
from ray.train.sklearn._sklearn_utils import _set_cpu_params
from ray.util.joblib import register_ray

from ray.train.sklearn import SklearnPredictor

class SklearnPredictProba(SklearnPredictor):
    '''
    Custom class based on Ray's SklearnPredictor
    Use .predict_proba() instead of .predict() to use with Scikit-Learn models

    '''
    def __init__(self, estimator, preprocessor = None):
        super().__init__(estimator, preprocessor)

    def _predict_pandas(self, data, feature_columns = None, num_estimator_cpus = 1, **predict_kwargs):
        register_ray()

        if num_estimator_cpus:
            _set_cpu_params(self.estimator, num_estimator_cpus)

        if TENSOR_COLUMN_NAME in data:
            data = data[TENSOR_COLUMN_NAME].to_numpy()
            data = _unwrap_ndarray_object_type_if_needed(data)
            if feature_columns:
                data = data[:, feature_columns]
        elif feature_columns:
            data = data[feature_columns]

        with parallel_backend("ray", n_jobs=num_estimator_cpus):
            df = pd.DataFrame(self.estimator.predict_proba(data, **predict_kwargs))

        df.columns = (
            ["predictions"]
            if len(df.columns) == 1
            else [f"{i}" for i in range(len(df.columns))]
        )
        return df
