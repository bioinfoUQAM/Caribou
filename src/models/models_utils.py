import os
import warnings
import numpy as np
import pandas as pd

# Class construction
from abc import ABC, abstractmethod

# Class weights
from sklearn.utils.class_weight import compute_class_weight

__author__ = 'Nicolas de Montigny'

__all__ = ['ModelsUtils']

# Ignore warnings to have a more comprehensible output on stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

class ModelsUtils(ABC):
    """
    Utilities for preprocessing data and doing cross validation using ray

    ----------
    Attributes
    ----------

    k : int
        The length of K-mers extracted

    classifier : string
        The name of the classifier to be used

    outdir : string
        Path to a folder to output results

    batch_size : int
        Size of the batch used for online learning

    taxa : string
        The taxa for which the model is trained in classifying

    ----------
    Methods
    ----------

    fit : only train or cross-validate training of classifier
        X : ray.data.Dataset
            Dataset containing the K-mers profiles of sequences for learning
        y : ray.data.Dataset
            Dataset containing the classes of sequences for learning

    predict : abstract method to predict the classes of a dataset

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
        # Parameters
        self.classifier = classifier
        self.batch_size = batch_size
        self.taxa = taxa
        self.kmers = kmers_list
        # Initialize hidden
        self._csv = csv
        self._nb_kmers = len(kmers_list)
        self._training_epochs = training_epochs
        # Initialize empty
        self._clf = None
        self._weights = {}
        self._scaler = None
        self._encoder = None
        self._trainer = None
        self._reductor = None
        self._predictor = None
        self._labels_map = None
        self._model_ckpt = None
        self._train_params = {}
        self._preprocessor = None
        self._workdir = outdir_model

    @abstractmethod
    def preprocess(self, ds):
        """
        """

    @abstractmethod
    def fit(self):
        """
        """

    @abstractmethod
    def predict(self):
        """
        """

    @abstractmethod
    def _prob_2_cls(self):
        """
        """

    @abstractmethod
    def _label_decode(self):
        """
        """

    def _compute_weights(self):
        """
        Set class weights depending on their abundance in data-associated classes csv
        """
        cls = pd.read_csv(self._csv)
        classes = list(cls[self.taxa].unique())
        weights = compute_class_weight(
            class_weight = 'balanced',
            classes = classes,
            y = cls[self.taxa]
        )
        for lab, encoded in self._labels_map:
            self._weights[encoded] = weights[classes.index(lab)]