
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List
from warnings import warn
from os.path import isfile
from ray.data import Dataset
from utils import save_Xy_data, load_Xy_data

from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import DictionaryLearning

from ray.data.preprocessor import Preprocessor
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

TENSOR_COLUMN_NAME = '__value__'

class TensorTruncatedSVDDecomposition(Preprocessor):
    """
    Custom class for using a mix of TruncatedSVD inspired by sklearn.decomposition.TruncatedSVD and applying a batched strategy inspired by sklearn.decomposition.IncrementalPCA to process batches sequentially.
    This makes it possible to use the class as a Ray preprocessor in a features reduction strategy.
    TruncatedSVD performs linear dimensionality reduction by means of truncated singular value decomposition (SVD).
    When it is applied following the TF-IDF normalisation, it becomes a latent semantic analysis (LSA).
    https://scikit-learn.org/stable/modules/decomposition.html#truncated-singular-value-decomposition-and-latent-semantic-analysis
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA
    """
    def __init__(self, features: List[str], nb_components: int = 10000, file: str = ''):
        # Parameters
        self.features = features
        self._nb_features = len(features)
        self._nb_components = nb_components
        self._file = file
        
    def _fit(self, ds: Dataset) -> Preprocessor:
        """
        TODO: adapt by using the metho from PySpark SVD
        https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.linalg.distributed.SingularValueDecomposition.html?highlight=svd

        Possibilities for parallel TruncatedSVD
        * sklearn minibatch PCA -> PCA / SVD mostly equivalent
        * implement parallel based on other library
        * dask-ml has a truncated svd
        * tf has a svd function
        """
        """
        Option to implement parallel computation for SVD
        1- Sparse Dictionnary Learning -> encode data to sparse representation by sample
        2- Sparse PCA (sparse SVD?) -> construct a PCA from sparsely encoded data
        It is possible to parallelize batches computation by applying the logic from MiniBatchDictionaryLearning and MiniBatchSparsePCA
        """
        # Parallel
        
        def batch_svd(batch):
            batch = batch[TENSOR_COLUMN_NAME]
            batch = _unwrap_ndarray_object_type_if_needed(batch)
            U, S, V = randomized_svd(
                batch,
                n_components = self._nb_components,
                n_iter = 2,
                power_iteration_normalizer = 'LU',
            )
            return {'V' : V}

        components = []
        if self._nb_features > self._nb_components:
            if isfile(self._file):
                components = np.array(load_Xy_data(self._file))
            else:
                # sampl = ds.random_sample(0.1)
                # svd = sampl.map_batches(batch_svd, batch_format = 'numpy')
                svd = ds.map_batches(batch_svd, batch_format = 'numpy')
                components = svd.limit(self._nb_components).to_pandas()['V']
                components = _unwrap_ndarray_object_type_if_needed(components)

                save_Xy_data(components, self._file)

            self.stats_ = {'components' : components}
        else:
            warn('No features reduction to do because the number of features is already lower than the required number of components')
            self.stats_ = {'components' : False}
        """
        # Parallel multiple MiniBatchDictionaryLearning
        def batch_svd(batch):
            batch = batch[TENSOR_COLUMN_NAME]
            batch = _unwrap_ndarray_object_type_if_needed(batch)
            dict = DictionaryLearning(
                n_components = self._nb_components,
                max_iter = 10,
                transform_algorithm = 'lasso_lars',
            )
            dict.fit(batch)
            return {'dictonnary' : [dict.components_]}
        components = []
        if self._nb_features > self._nb_components:
            if isfile(self._file):
                components = np.array(load_Xy_data(self._file))
            else:
                svd = ds.map_batches(batch_svd, batch_format = 'numpy')
                print(svd.to_pandas())
                for row in svd.iter_rows():
                    components.append(row['dictonnary'])
                components = np.mean(components, axis = 0)
                print(components.shape)
                save_Xy_data(components, self._file)
        """
        # Incremental
        # components = None
        # singular_values = None
        # if self._nb_features > self._nb_components:
        #     for batch in tqdm(ds.iter_batches(batch_format = 'numpy')):
        #         batch = batch[TENSOR_COLUMN_NAME]
        #         batch = _unwrap_ndarray_object_type_if_needed(batch)
        #         if components is not None:
        #             # Build matrix of previous computations
        #             batch = np.vstack(
        #                 (
        #                     singular_values.reshape((-1, 1)) * components,
        #                     batch,
        #                 )
        #             )
        #         # U : (1000, 100), S : (100,), V : (100, 1024)
        #         # S.reshape : (100, 1), S.reshape * components : (100, 1024)
        #         # batch : (1000, 1024), vstack : (1100, 1024)
        #         U, Sigma, VT = randomized_svd(
        #             batch,
        #             n_components = self._nb_components,
        #             n_iter = 1,
        #             power_iteration_normalizer = 'LU',
        #         )
        #         components = VT
        #         singular_values = Sigma
                

        return self

    def _transform_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        # _validate_df(df, TENSOR_COLUMN_NAME, self._nb_features)
        components = self.stats_['components']
        
        if components is not False:
            tensor_col = df[TENSOR_COLUMN_NAME]
            tensor_col = _unwrap_ndarray_object_type_if_needed(tensor_col)
            tensor_col = np.dot(tensor_col, components.T)
            df[TENSOR_COLUMN_NAME] = pd.Series(list(tensor_col))

        return df

    def __repr__(self):
        return (f"{self.__class__.__name__}(features={self._nb_features!r}, file={self._file!r})")

def _validate_df(df: pd.DataFrame, column: str, nb_features: int) -> None:
    if len(df.loc[0, column]) != nb_features:
        raise ValueError('Discordant number of features in the tensor column with the one from the dataframe used for fitting')
