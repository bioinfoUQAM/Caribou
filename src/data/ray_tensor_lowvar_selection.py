
import numpy as np
import pandas as pd

from typing import List
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor

class TensorLowVarSelection(Preprocessor):
    """
    Custom implementation of VarianceThreshold inspired by sklearn.feature_selection.VarianceThreshold features selector to be used as a Ray preprocessor.
    """
    def __init__(
        self,
        tensor_column : str,
        features_list : List[str],
        threshold: float = np.inf,
        nb_keep : int = np.inf,
    ):
        self.column = tensor_column
        self.features_list = features_list
        if 'id' in self.features_list:
            self.features_list.remove('id')
        self.nb_features = len(self.features_list)
        self.threshold = threshold
        self.nb_keep = nb_keep
        self.stats_ = []
        self.removed_features = []

    def _fit(self, ds: Dataset) -> Preprocessor:
        nb_records = ds.count()
        #
        sum_arr = np.zeros(self.nb_features)
        mean_arr = np.zeros(self.nb_features)
        sqr_dev_arr = np.zeros(self.nb_features)
        var_arr = np.zeros(self.nb_features)
        #
        def sum_func(arr, sum_arr):
            return np.add(sum_arr, np.sum(arr, axis=0))

        def mean_func(arr, nb_records):
            return np.divide(arr, nb_records)

        def sqr_dev_func(arr, mean_arr, sqr_dev_arr):
            return np.add(sqr_dev_arr, np.sum(np.power(np.subtract(arr, mean_arr), 2), axis = 0))

        if self.nb_keep != np.inf or self.threshold != np.inf:
            # Get sum per column
            for batch in ds.iter_batches(
                batch_size = 100,
                batch_format = 'numpy'
            ):
                sum_arr = sum_func(batch, sum_arr)
            # Get mean per column
            mean_arr = mean_func(sum_arr, nb_records)
            # Get sum of deviation
            for batch in ds.iter_batches(
                batch_size = 100,
                batch_format = 'numpy'
            ):
                sqr_dev_arr = sqr_dev_func(batch, mean_arr, sqr_dev_arr)
            # Get variance per column
            var_arr = mean_func(sqr_dev_arr, nb_records)
            p10 = 0.1 * self.nb_features

            if self.nb_keep != np.inf and (self.nb_keep + (p10 * 2)) < self.nb_features:
                var_mapping = {ind : var_arr[ind] for ind in np.arange(self.nb_features)}
                keep_arr = np.ravel(np.sort(var_arr))
                keep_arr = keep_arr[p10:len(keep_arr)-p10]
                keep_arr = np.random.choice(keep_arr, self.nb_keep)
                remove_arr = np.ravel(np.sort(var_arr))
                remove_arr = np.array([ind for ind in remove_arr if ind not in keep_arr])

                # Switch values from keep_arr to remove if number is discordant
                if len(keep_arr) > self.nb_keep:
                    nb_switch = len(keep_arr) - self.nb_keep
                    remove_arr = np.insert(remove_arr, 0, keep_arr[:nb_switch])
                    keep_arr = keep_arr[nb_switch:]
                elif len(keep_arr) < self.nb_keep:
                    nb_switch = self.nb_keep - len(keep_arr)
                    keep_arr = np.insert(keep_arr, 0, remove_arr[nb_switch:])
                    remove_arr = remove_arr[:nb_switch]
                # Loop to assign values to remove
                for k, v in var_mapping.items():
                    if v in remove_arr:
                        pos_v = int(np.where(remove_arr == v)[0][0])
                        remove_arr = np.delete(remove_arr, pos_v)
                        self.stats_.append(k)
                self.removed_features = [self.features_list[ind] for ind in self.stats_]

            elif self.threshold != np.inf:
                for ind in np.arange(self.nb_features):
                    variance = var_arr[ind]
                    if variance <= self.threshold:
                        self.stats_.append(ind)
                self.removed_features = [self.features_list[ind] for ind in self.stats_]

        return self

    def _transform_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(self.stats_) > 0 :
            _validate_df(df, self.column, self.nb_features)
            df_out = pd.DataFrame(columns = [self.column])

            for ind, row in enumerate(df.iterrows()):
                tensor = np.delete(row[1].to_numpy()[0], self.stats_, axis=0)
                df_out.loc[ind, self.column] = tensor
            
            return df_out        
        else:
            return df

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(threshold={self.threshold!r}, nb_keep={self.nb_keep!r})"
        )


def _validate_df(df: pd.DataFrame, column: str, nb_features: int) -> None:
    if len(df.loc[0, column]) != nb_features:
        raise ValueError('Discordant number of features in the tensor column with the one from the dataframe used for fitting')