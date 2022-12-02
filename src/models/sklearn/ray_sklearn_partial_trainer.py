import os
import ray
import warnings
from time import time
from traceback import format_exc
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.metrics import check_scoring

# we are using a private API here, but it's consistent across versions
from sklearn.model_selection._validation import _check_multimetric_scoring, _score

from ray import tune
import ray.cloudpickle as cpickle
from ray.air._internal.checkpointing import (
    save_preprocessor_to_dir,
)
from ray.util.joblib import register_ray
from ray.air.config import RunConfig, ScalingConfig
from ray.train.constants import MODEL_KEY, TRAIN_DATASET_KEY
from ray.train.sklearn._sklearn_utils import _set_cpu_params

from ray.train.sklearn import SklearnTrainer

class SklearnPartialTrainer(SklearnTrainer):
    """
    Class adapted from Ray's SklearnTrainer class to allow for partial_fit and usage of tensors as inputs.
    """

    def __init__(
        self,
        *,
        estimator,
        datasets,
        label_column = None,
        labels_list = None,
        features_list = None,
        params = None,
        scoring = None,
        cv = None,
        return_train_score_cv = False,
        parallelize_cv = None,
        set_estimator_cpus = True,
        scaling_config = None,
        run_config = None,
        preprocessor = None,
        batch_size = 32,
        **fit_params
    ):
        super().__init__(
        estimator = estimator,
        datasets = datasets,
        label_column = label_column,
        params = params,
        scoring = scoring,
        cv = cv,
        return_train_score_cv = return_train_score_cv,
        parallelize_cv = parallelize_cv,
        set_estimator_cpus = set_estimator_cpus,
        scaling_config = scaling_config,
        run_config = run_config,
        preprocessor = preprocessor,
        **fit_params
        )
        self._batch_size = batch_size
        self._labels = labels_list
        self._features_list = features_list

    def _validate_attributes(self):
        # Run config
        if not isinstance(self.run_config, RunConfig):
            raise ValueError(
                f"`run_config` should be an instance of `ray.air.RunConfig`, "
                f"found {type(self.run_config)} with value `{self.run_config}`."
            )
        # Scaling config
        if not isinstance(self.scaling_config, ScalingConfig):
            raise ValueError(
                "`scaling_config` should be an instance of `ScalingConfig`, "
                f"found {type(self.scaling_config)} with value `{self.scaling_config}`."
            )
        # Datasets
        if not isinstance(self.datasets, dict):
            raise ValueError(
                f"`datasets` should be a dict mapping from a string to "
                f"`ray.data.Dataset` objects, "
                f"found {type(self.datasets)} with value `{self.datasets}`."
            )
        # Preprocessor
        if self.preprocessor is not None and not isinstance(
            self.preprocessor, ray.data.Preprocessor
        ):
            raise ValueError(
                f"`preprocessor` should be an instance of `ray.data.Preprocessor`, "
                f"found {type(self.preprocessor)} with value `{self.preprocessor}`."
            )

        if self.resume_from_checkpoint is not None and not isinstance(
            self.resume_from_checkpoint, ray.air.Checkpoint
        ):
            raise ValueError(
                f"`resume_from_checkpoint` should be an instance of "
                f"`ray.air.Checkpoint`, found {type(self.resume_from_checkpoint)} "
                f"with value `{self.resume_from_checkpoint}`."
            )


        if self.label_column is not None and not isinstance(self.label_column, str):
            raise ValueError(
                f"`label_column` must be a string or None, got '{self.label_column}'"
            )

        if self.params is not None and not isinstance(self.params, dict):
            raise ValueError(f"`params` must be a dict or None, got '{self.params}'")

        # Don't validate self.scoring for now as many types are supported
        # Don't validate self.cv for now as many types are supported

        if not isinstance(self.return_train_score_cv, bool):
            raise ValueError(
                f"`return_train_score_cv` must be a boolean, got "
                f"'{self.return_train_score_cv}'"
            )

        if TRAIN_DATASET_KEY not in self.datasets:
            raise KeyError(
                f"'{TRAIN_DATASET_KEY}' key must be preset in `datasets`. "
                f"Got {list(self.datasets.keys())}"
            )
        if "cv" in self.datasets:
            raise KeyError(
                "'cv' is a reserved key. Please choose a different key "
                "for the dataset."
            )
        if (
            not isinstance(self.parallelize_cv, bool)
            and self.parallelize_cv is not None
        ):
            raise ValueError(
                "`parallelize_cv` must be a bool or None, got "
                f"'{self.parallelize_cv}'"
            )
        scaling_config = self._validate_scaling_config(self.scaling_config)
        if (
            self.cv
            and self.parallelize_cv
            and scaling_config.trainer_resources.get("GPU", 0)
        ):
            raise ValueError(
                "`parallelize_cv` cannot be True if there are GPUs assigned to the "
                "trainer."
            )

    def _get_datasets(self):
        out_datasets = {}
        for key, ray_dataset in self.datasets.items():
            ray_dataset = ray.get(ray_dataset)
            out_datasets[key] = (
                ray_dataset.drop_columns([self.label_column]),
                ray.data.from_pandas(pd.DataFrame(ray_dataset.to_pandas()[self.label_column], columns = [self.label_column])),
            )
        return out_datasets

    def training_loop(self):
        register_ray()

        self.estimator.set_params(**self.params)

        datasets = self._get_datasets()

        X_train, y_train = datasets.pop(TRAIN_DATASET_KEY)

        scaling_config = self._validate_scaling_config(self.scaling_config)

        num_workers = scaling_config.num_workers or 0
        assert num_workers == 0  # num_workers is not in scaling config allowed_keys

        num_cpus = int(-1)

        # see https://scikit-learn.org/stable/computing/parallelism.html
        os.environ["OMP_NUM_THREADS"] = str(num_cpus)
        os.environ["MKL_NUM_THREADS"] = str(num_cpus)
        os.environ["OPENBLAS_NUM_THREADS"] = str(num_cpus)
        os.environ["BLIS_NUM_THREADS"] = str(num_cpus)

        _set_cpu_params(self.estimator, num_cpus)

        with parallel_backend("ray", n_jobs=num_cpus):
            start_time = time()
            for batch_X, batch_y in zip(
                X_train.iter_batches(
                    batch_size = self._batch_size,
                    batch_format = 'numpy'
                ),
                y_train.iter_batches(
                    batch_size = self._batch_size,
                    batch_format = 'numpy'
                )
            ):  
                try:
                    batch_X = pd.DataFrame(batch_X, columns = self._features_list)
                except:
                    for i in range(len(batch_X)):
                        if len(batch_X[i]) != len(self._features_list):
                            warnings.warn("The features list length for some reads are not the same as for other reads.\
                                Removing the last {} additionnal values, this may influence training.\
                                    If error persists over multiple samples, please rerun the K-mers extraction".format(len(batch_X[i]) - len(self._features_list)))
                            batch_X[i] = batch_X[i][:len(self._features_list)]
                try:
                    self.estimator.partial_fit(batch_X, np.ravel(batch_y), classes = self._labels, **self.fit_params)
                except TypeError:
                    self.estimator.partial_fit(batch_X, np.ravel(batch_y), **self.fit_params)
            fit_time = time() - start_time

            with tune.checkpoint_dir(step=1) as checkpoint_dir:
                with open(os.path.join(checkpoint_dir, MODEL_KEY), "wb") as f:
                    cpickle.dump(self.estimator, f)

                if self.preprocessor:
                    save_preprocessor_to_dir(self.preprocessor, checkpoint_dir)

            if self.label_column:
                validation_set_scores = self._score_on_validation_sets(
                    self.estimator, datasets
                )
                cv_scores = {}
            else:
                validation_set_scores = {}
                cv_scores = {}

        # cv_scores will not override validation_set_scores as we
        # check for that during initialization
        results = {
            **validation_set_scores,
            **cv_scores,
            "fit_time": fit_time,
        }
        tune.report(**results)

    def _score_on_validation_sets(
        self,
        estimator,
        datasets
    ):
        results = defaultdict(dict)
        if not datasets:
            return results

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(estimator, self.scoring)

        for key, X_y_tuple in datasets.items():
            X_test, y_test = X_y_tuple

            for batch in X_test.iter_batches(
                batch_size = X_test.count(),
                batch_format = 'numpy'
            ):
                X_test = pd.DataFrame(batch, columns = self._features_list)
            start_time = time()
            try:
                test_scores = _score(estimator, X_test, y_test.to_pandas(), scorers)
            except Exception:
                if isinstance(scorers, dict):
                    test_scores = {k: np.nan for k in scorers}
                else:
                    test_scores = np.nan
                warnings.warn(
                    f"Scoring on validation set {key} failed. The score(s) for "
                    f"this set will be set to nan. Details: \n"
                    f"{format_exc()}",
                    UserWarning,
                )
            score_time = time() - start_time
            results[key]["score_time"] = score_time
            if not isinstance(test_scores, dict):
                test_scores = {"score": test_scores}

            for name in test_scores:
                results[key][f"test_{name}"] = test_scores[name]
        return results
