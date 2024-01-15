import numpy as np
import pandas as pd

from typing import List
from ray.data import Dataset

from xgboost import XGBRFClassifier

from sklearn.preprocessing import LabelEncoder


from ray.data.preprocessor import Preprocessor
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

TENSOR_COLUMN_NAME = '__value__'

class TensorRDFFeaturesSelection(Preprocessor):
    """
    Wrapper class for using Random Forest Classifier from XGBoost in features selection as a Ray preprocessor.
    XGBRFClassifier trains a random forest of decision trees that is used to determine the features that are most useful in classification.
    https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
    """
    
    def __init__(self, features: List[str], taxa: str):
        # Parameters
        self.taxa = taxa
        self.features = features
        self._nb_features = len(features)

    def _fit(self, ds: Dataset) -> Preprocessor:
        def xgboost_batch(arr: np.array):
            # Labels data
            y = arr[self.taxa]
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
            # Features data
            X = _unwrap_ndarray_object_type_if_needed(arr[TENSOR_COLUMN_NAME])
            X = pd.DataFrame(X, columns = self.features)
            # XGBoost tree
            tree = XGBRFClassifier()
            tree.fit(X,y)
            # Used features in the tree
            tree = tree.get_booster()
            relevant_features = tree.get_fscore()
            relevant_features = [feat for feat in relevant_features.keys()]

            return {'features':[relevant_features]}
        
        cols_keep = []

        relevant_features = ds.map_batches(xgboost_batch, batch_format = 'numpy')
        for row in relevant_features.iter_rows():
            cols_keep.extend(row['features'])
        cols_keep = np.unique(cols_keep)

        if 0 < len(cols_keep) :
            self.stats_ = {'cols_keep' : cols_keep}
        else:
            self.stats_ = {'cols_keep' : self.features}

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
        return (f"{self.__class__.__name__}(features={self._nb_features!r}, taxa={self.taxa!r}, threshold={self.threshold!r})")

def _validate_df(df: pd.DataFrame, column: str, nb_features: int) -> None:
    if len(df.loc[0, column]) != nb_features:
        raise ValueError('Discordant number of features in the tensor column with the one from the dataframe used for fitting')