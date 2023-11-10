import os
import warnings
import pandas as pd

# Class construction
from abc import ABC, abstractmethod

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
        kmers_list
    ):
        # Parameters
        self.classifier = classifier
        self.batch_size = batch_size
        self.taxa = taxa
        self.kmers = kmers_list
        # Initialize hidden
        self._nb_kmers = len(kmers_list)
        self._training_epochs = training_epochs
        # Initialize empty
        self._labels_map = None
        # Initialize Ray variables
        self._clf = None
        self._encoder = None
        self._scaler = None
        self._preprocessor = None
        self._reductor = None
        self._model_ckpt = None
        self._trainer = None
        self._train_params = {}
        self._predictor = None
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