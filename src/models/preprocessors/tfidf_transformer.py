
import numpy as np
import pandas as pd
import scipy.sparse as sp


from ray.data.dataset import Dataset
from sklearn.preprocessing import normalize
from ray.data.preprocessor import Preprocessor
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

TENSOR_COLUMN_NAME = '__value__'

class TensorTfIdfTransformer(Preprocessor):
    """
    Custom implementation of TF-IDF transformation inspired by sklearn.feature_extraction.text.TfidfTransformer features scaler to be used as a Ray preprocessor.
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer
    TF-IDF transformation is used to scale down the impact of tokens that occur very frequently and scale up the impact of those that occur very rarely.
    """

    def __init__(self, features):
        # Parameters
        self._features = features
        self._nb_features = len(features)

    def _fit(self, ds: Dataset) -> Preprocessor:
        nb_samples = ds.count()

        # Nb of occurences
        def get_occurences(batch):
            batch = batch[TENSOR_COLUMN_NAME]
            return {'occurences' : np.count_nonzero(batch, axis = 0)}

        # Nb of occurences
        occurences = np.zeros(self._nb_features)
        # occur = ds.map_batches(get_occurences, batch_format = 'numpy')
        for batch in ds.iter_batches(batch_format = 'numpy'):
            batch = batch[TENSOR_COLUMN_NAME]
            occurences += np.count_nonzero(batch, axis = 0)
        # for row in occur.iter_rows():
        #     occurences += row['occurences']

        idf = np.log(nb_samples / occurences) + 1
        
        idf_diag = sp.diags(
            idf,
            offsets=0,
            shape=(self._nb_features, self._nb_features),
            format="csr",
            dtype=np.float64,
        )
        
        self.stats_ = {'idf_diag' : idf_diag}

        return self
    
    def _transform_pandas(self, batch: pd.DataFrame) -> pd.DataFrame:
        # _validate_df(batch, TENSOR_COLUMN_NAME, self._nb_features)
        idf_diag = self.stats_['idf_diag']
        
        df = batch[TENSOR_COLUMN_NAME]
        df = _unwrap_ndarray_object_type_if_needed(df)

        df = df * idf_diag
        
        df = normalize(df, norm = 'l2', copy = False)

        batch[TENSOR_COLUMN_NAME] = pd.Series(list(df))

        return batch

def _validate_df(df: pd.DataFrame, column: str, nb_features: int) -> None:
    if len(df.loc[0, column]) != nb_features:
        raise ValueError('Discordant number of features in the tensor column with the one from the dataframe used for fitting')