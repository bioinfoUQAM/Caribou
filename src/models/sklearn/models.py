import os
import warnings

import numpy as np
import pandas as pd

# Class construction
from abc import ABC, abstractmethod

# Parent class
from models.models_utils import ModelsUtils

TENSOR_COLUMN_NAME = '__value__'
LABELS_COLUMN_NAME = 'labels'

__author__ = 'Nicolas de Montigny'

__all__ = ['SklearnModel']

# Ignore warnings to have a more comprehensible output on stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

class SklearnModels(ModelsUtils, ABC):
    """
    Class used to build, train and predict models using Ray with Scikit-learn backend

    ----------
    Attributes
    ----------

    clf_file : string
        Path to a file containing the trained model for this object

    ----------
    Methods
    ----------

    preprocess : preprocess the data before training and splitting the original dataset in case of cross-validation

    train : train a model using the given datasets

    predict : predict the classes of a dataset
        ds : ray.data.Dataset
            Dataset containing K-mers profiles of sequences to be classified

        threshold : float
            Minimum percentage of probability to effectively classify.
            Sequences will be classified as 'unknown' if the probability is under this threshold.
            Defaults to 80%
    """
    def __init__(
        self,
        classifier,
        outdir_model,
        batch_size,
        training_epochs,
        taxa,
        kmers_list,
        csv
    ):
        super().__init__(
            classifier,
            outdir_model,
            batch_size,
            training_epochs,
            taxa,
            kmers_list,
            csv
        )
        
    @abstractmethod
    def preprocess(self):
        """
        """
        
    @abstractmethod
    def _build(self):
        """
        """

    @abstractmethod
    def fit(self, datasets):
        """
        """

    @abstractmethod
    def predict(self, ds):
        """
        """

    @abstractmethod
    def predict_proba(self):
        """
        """

    @abstractmethod
    def _get_threshold_pred(self):
        """
        """

    def _label_decode(self, predict):
        print('_label_decode')
        decoded = pd.Series(np.empty(len(predict), dtype=object))
        for label, encoded in self._labels_map.items():
            decoded[predict == encoded] = label

        return np.array(decoded)