import os
import warnings
import pandas as pd

# Class construction
from abc import ABC, abstractmethod

# CV metrics
from sklearn.metrics import precision_recall_fscore_support

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

    train : only train or cross-validate training of classifier
        X : ray.data.Dataset
            Dataset containing the K-mers profiles of sequences for learning
        y : ray.data.Dataset
            Dataset containing the classes of sequences for learning
        cv : boolean
            Should cross-validation be verified or not.
            Defaults to True.

    predict : abstract method to predict the classes of a dataset

    """
    def __init__(
        self,
        classifier,
        dataset,
        outdir_model,
        outdir_results,
        batch_size,
        training_epochs,
        k,
        taxa,
        kmers_list,
        verbose
    ):
        # Parameters
        self.classifier = classifier
        self.dataset = dataset
        self.outdir_results = outdir_results
        self.batch_size = batch_size
        self.k = k
        self.taxa = taxa
        self.kmers = kmers_list
        self.verbose = verbose
        # Initialize hidden
        self._nb_kmers = len(kmers_list)
        self._training_epochs = training_epochs
        # Initialize empty
        self._labels_map = None
        self._predict_ids = []
        # Initialize Ray variables
        self._clf = None
        self._preprocessor = None
        self._model_ckpt = None
        self._trainer = None
        self._train_params = {}
        self._predictor = None
        self._workdir = outdir_model
        # Files
        self._cv_csv = os.path.join(self.outdir_results,'{}_{}_K{}_cv_scores.csv'.format(self.classifier, self.taxa, self.k))

    @abstractmethod
    def preprocess(self, df):
        """
        """

    @abstractmethod
    def train(self):
        """
        """

    @abstractmethod
    def _fit_model(self):
        """
        """

    @abstractmethod
    def _cross_validation(self):
        """
        """

    def _cv_score(self, y_true, y_pred):
        print('_cv_score')

        y_compare = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred
        })
        y_compare['y_true'] = y_compare['y_true'].str.lower()
        y_compare['y_pred'] = y_compare['y_pred'].str.lower()
        y_compare.to_csv(os.path.join(self._workdir, f'y_compare_{self.dataset}_{self.classifier}.csv'))

        support = precision_recall_fscore_support(
            y_compare['y_true'],
            y_compare['y_pred'],
            average = 'weighted'
        )

        scores = pd.DataFrame(
            {self.classifier : [support[0],support[1],support[2]]},
            index = ['Precision','Recall','F-score']
        )

        scores.to_csv(self._cv_csv, index = True)

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