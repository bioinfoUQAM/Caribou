
import numpy as np
import pandas as pd

from typing import List
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

TENSOR_COLUMN_NAME = '__value__'

class TensorLowVarSelection(Preprocessor):
    """
    Custom implementation of VarianceThreshold inspired by sklearn.feature_selection.VarianceThreshold features selector to be used as a Ray preprocessor.
    https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold
    """
    def __init__(
        self,
        features : List[str],
        threshold: float = 0.1,
    ):
        self.features = features
        self.threshold = threshold
        self._nb_features = len(features)

    def _fit(self, ds: Dataset) -> Preprocessor:
        cols_keep = []
        nb_samples = ds.count()
        sum_arr = np.zeros(self._nb_features)
        mean_arr = np.zeros(self._nb_features)
        sqr_dev_arr = np.zeros(self._nb_features)
        var_arr = np.zeros(self._nb_features)
        
        # Function for parallel sum computing
        def get_sums(batch):
            df = batch[TENSOR_COLUMN_NAME]
            df = _unwrap_ndarray_object_type_if_needed(df)
            return({'sum' : [np.sum(df, axis = 0)]})
        
        # Sum per column
        sums = ds.map_batches(get_sums, batch_format = 'pandas')
        for row in sums.iter_rows():
            sum_arr += row['sum']
        
        # Mean per column
        mean_arr = sum_arr / nb_samples
        
        # Function for parallel squared deviation computing
        def get_sqr_dev(batch):
            df = batch[TENSOR_COLUMN_NAME]
            df = _unwrap_ndarray_object_type_if_needed(df)
            return({'sqr_dev' : [np.sum(np.power(np.subtract(df, mean_arr), 2), axis = 0)]})
        
        # Sum of deviation per column
        sqr_devs = ds.map_batches(get_sqr_dev, batch_format = 'pandas')
        for row in sqr_devs.iter_rows():
            sqr_dev_arr += row['sqr_dev']

        # Variance per column
        var_arr = sqr_dev_arr / nb_samples
        
        # Compute the threshold from distribution of variance values
        self.threshold = np.nanquantile(var_arr, self.threshold)

        # Keep features with values higher than the threshold
        cols_keep = [self.features[i] for i, var in enumerate(var_arr) if var > self.threshold]
        
        self.stats_ = {'cols_keep' : cols_keep}

        return self

    def _transform_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        # _validate_df(df, TENSOR_COLUMN_NAME, self._nb_features)
        cols_keep = self.stats_['cols_keep']

        if len(cols_keep) < self._nb_features:
            tensor_col = df[TENSOR_COLUMN_NAME]
            tensor_col = _unwrap_ndarray_object_type_if_needed(tensor_col)
            tensor_col = pd.DataFrame(tensor_col, columns = self.features)

            tensor_col = tensor_col[cols_keep].to_numpy()

            df[TENSOR_COLUMN_NAME] = pd.Series(list(tensor_col))        

        return df

    def __repr__(self):
        return (f"{self.__class__.__name__}(features={self._nb_features!r}, threshold={self.threshold!r})")

def _validate_df(df: pd.DataFrame, column: str, nb_features: int) -> None:
    if len(df.loc[0, column]) != nb_features:
        raise ValueError('Discordant number of features in the tensor column with the one from the dataframe used for fitting')