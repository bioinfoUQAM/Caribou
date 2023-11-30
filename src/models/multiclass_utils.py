import os
import ray
import warnings
import numpy as np
import pandas as pd

# Class construction
from abc import ABC, abstractmethod

from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

__author__ = 'Nicolas de Montigny'

__all__ = ['ModelsUtils']

TENSOR_COLUMN_NAME = '__value__'

class MulticlassUtils(ABC):
    """
    Abstract class to provide utilities for multiclass classification models.
    These methods are meant to be used when decomposing data into taxonomic groups before training one model per group
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

    def _split_dataset(self, ds, taxa, csv):
        """
        Splits the dataset's taxa column into a collection of smaller datasets according to the previous taxonomic level labels
        
        Makes assumption that classes are order specific -> broad in csv columns

        Ray data GroupBy https://www.anyscale.com/blog/training-one-million-machine-learning-models-in-record-time-with-ray#approach-2:-using-ray-data-(grouping-data-by-key)
        1. GroupBy previous taxa
        2. Fx for model training (train_fx)
        3. ds.map_groups(train_fx) to exec the training of models in parallel
        4. Write results to file / save models
        """
        ds_collection = {}
        # cls = pd.read_csv(csv)
        # prev_tax = list(cls.columns)
        # prev_tax = prev_tax[prev_tax.index(taxa) + 1]
        # unique_labs = cls[prev_tax].unique()


        # for lab in unique_labs:
            
        # def map_split(ds):
        #     logging.getLogger("ray").info(ds[ds[prev_tax] == lab])
        #     return ds[ds[prev_tax] == lab]

        # test = ds.map(map_split)

        # partial_ds = ds.map_batches(map_split, batch_format = 'pandas')
        # file = '/home/nick/github/test'
        # partial_ds.write_parquet(file)
        # ds_collection[lab] = partial_ds

        # for k, v in ds_collection.items():
        #     # print(v.to_pandas())
        #     print(v)
        """
        for lab in unique_labs:
            ds_collection[lab] = []

        for batch in ds.iter_batches(batch_format = 'pandas'):
            labs_batch = batch[prev_tax].unique()
            for lab in labs_batch:
                ds_collection[lab].append(batch[batch[prev_tax] == lab])

        for lab in unique_labs:
            ds_collection[lab] = pd.concat(ds_collection[lab])
        """
        return ds_collection

    def _predictions_cv(self, predictions):
        """
        Brings back together the predictions made by multiple models trained on subclasses of the original dataset
        
        If multiple sub-models classify a sample with same probability, use a soft voting logic to determine which one to classify to
        
        ----------
        Cross-validation
        ----------
        * We know the classes from the previous taxa, can make each model CV on their subpart
        * Metrics for CV overall per taxa ~k-fold strategy (mean / mode)
        """

    
    def _predictions_classif(self, predictions):
        """
        Brings back together the predictions made by multiple models trained on subclasses of the original dataset
        
        If multiple sub-models classify a sample with same probability, use a soft voting logic to determine which one to classify to
        
        ----------
        Classification
        ----------
        * Since we know the previous taxa classified per sequence, we can run this specific model to classify at the current level
        * See multi-stage classification
        """
