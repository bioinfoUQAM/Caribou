import os
import ray
import warnings
import numpy as np
import pandas as pd

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
    Ray data GroupBy
    -----------------------
    https://www.anyscale.com/blog/training-one-million-machine-learning-models-in-record-time-with-ray#approach-2:-using-ray-data-(grouping-data-by-key)
    1. GroupBy previous taxa
    2. Fx for model training (train_fx)
    3. ds.map_groups(train_fx) to exec the training of models in parallel
    4. Write results to file / save models
    
    -----------------------
    Mixture-of-Experts (MoE)
    -----------------------
    1. Train each expert on their task-associated data
        * Split training data into 80/20% splits
        * Train/val over multiple epochs
    2. Train a gating network on the whole task
        * Perceptron NN for gating
        * Train on whole training ds
        * Validation on simulated reads ds
        * CV on test simulated reads ds 
    https://medium.com/@bensalemh300/harnessing-the-best-of-both-worlds-how-mixture-of-experts-meets-pyspark-for-mnist-mastery-315f82e65a0e
    https://machinelearningmastery.com/mixture-of-experts/

    1. Cluster Data Split: Data within each cluster is divided into training and testing sets.
    2. Decision Tree Classifiers: For clusters where there’s more than one unique class in the training data, we train Decision Tree classifiers. These classifiers can distinguish between different classes within the cluster.
    3. Storing Expert Models: Trained Decision Tree models are stored in a dictionary, where each expert corresponds to a specific cluster.
    4. Performance Evaluation: The performance of each expert model is assessed by evaluating its accuracy on the corresponding test data.
    
    Sklearn LogisticRegression : https://github.com/zermelozf/esn-lm/blob/master/esnlm/readouts/smoe.py
    Keras/TF : https://abdulkaderhelwan.medium.com/mixture-of-experts-introduction-39f244a4ff05
    Keras/TF on article 2018 : https://github.com/drawbridge/keras-mmoe
    Keras/TF 2018 : https://github.com/eminorhan/mixture-of-experts
    Detailed example : https://mattgorb.github.io/moe
    Detailed example : https://towardsdatascience.com/how-to-build-a-wide-and-deep-model-using-keras-in-tensorflow-2-0-2f7a236b5a4b
    Keras example : https://keras.io/examples/nlp/text_classification_with_switch_transformer/
    Keras example : https://stackoverflow.com/questions/77551865/how-to-extend-keras-gpt2-model-moe-example
    FastMoE PyTorch : https://fastmoe.ai/
    Tutel PyTorch : https://www.microsoft.com/en-us/research/blog/tutel-an-efficient-mixture-of-experts-implementation-for-large-dnn-model-training/
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
        nb_clusters = int(ds.count() / 100)
        ds = ds.repartition(nb_clusters).add_column('cluster', lambda df: df.index % nb_clusters)
        return ds.groupby('cluster')
    
    