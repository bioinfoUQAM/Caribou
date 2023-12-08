import os
import ray
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa

# Class construction
from abc import ABC, abstractmethod
from models.models_utils import ModelsUtils

from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

__author__ = 'Nicolas de Montigny'

__all__ = ['ModelsUtils']

TENSOR_COLUMN_NAME = '__value__'

class MulticlassUtils(ModelsUtils, ABC):
    """
    Abstract class to provide utilities for multiclass classification models.
    
    These methods are meant to be used when decomposing data into taxonomic groups before training one model per group
    
    -----------------------
    Ray data GroupBy + Bagging meta-estimator
    -----------------------
    https://www.anyscale.com/blog/training-one-million-machine-learning-models-in-record-time-with-ray#approach-2:-using-ray-data-(grouping-data-by-key)
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
    1. GroupBy previous taxa
    2. Fx for model training (train_fx)
    3. ds.map_groups(train_fx) to exec the training of models in parallel
    4. Write results to file / save models
    """

    def _get_count_previous_taxa(self, taxa, csv):
        """
        Fetch the previous taxa and computes the number of classes in it

        Makes assumption that classes are ordered ``specific -> broad`` in csv columns
        
        Used to determine if the dataset should be splitted according to the previous taxonomic level labels
        """
        prev_taxa = None
        cls = pd.read_csv(csv)
        cols = list(cls.columns)
        prev_taxa = cols[cols.index(taxa) + 1]

        return prev_taxa, len(cls[prev_taxa].unique())

    def _prev_taxa_split_dataset(self, ds, prev_taxa = None):
        """
        Splits the dataset's taxa column into a collection of smaller datasets according to the previous taxonomic level labels
        """
        if prev_taxa is None:
            prev_taxa, nb_classes = self._get_count_previous_taxa(self.taxa,self._csv)
        return ds.groupby(prev_taxa)
    
    def _random_split_dataset(self, ds):
        """
        Assigns random numbers to a new column and group samples by it to form a collection of smaller random datasets
        
        Used when there is not enough labels in previous taxa for splitting according to the previous taxonomic level labels
        """
        def map_clusters(batch):
            clusters = np.arange(len(batch))
            batch['cluster'] = clusters
            return batch

        nb_clusters = int(ds.count() / 100)

        ds = ds.repartition(100)
        ds = ds.map_batches(map_clusters, batch_size = nb_clusters, batch_format = 'pandas')

        return ds.groupby('cluster')