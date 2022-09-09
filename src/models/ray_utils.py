import os
import ray
import warnings
import numpy as np
import pandas as pd

# Class construction
from abc import ABC, abstractmethod

# Data preprocessing
from ray.data.preprocessors import MinMaxScaler, LabelEncoder, Chain, SimpleImputer

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
    def __init__(self, classifier, outdir_results, batch_size, k, taxa, kmers_list, verbose):
        # Parameters
        self.classifier = classifier
        self.outdir_results = outdir_results
        self.batch_size = batch_size
        self.k = k
        self.taxa = taxa
        self.kmers = kmers_list
        self.verbose = verbose
        # Initialize hidden
        self._nb_kmers = len(kmers_list)
        # Initialize empty
        self._labels_map = None
        self._train_ids = []
        self._predict_ids = []
        # Initialize Ray variables
        self._clf = None
        self._model_ckpt = None
        self._preprocessor = None
        self._encoder = None
        self._trainer = None
        self._train_params = {}
        self._tuner = None
        self._tuning_params = {}
        self._predictor = None
        # Files
        self._cv_csv = os.path.join(self.outdir_results,'{}_{}_K{}_cv_scores.csv'.format(self.classifier, self.taxa, self.k))

    def _training_preprocess(self, X, y):
        print('_training_preprocess')
        for row in X.iter_rows():
            self._train_ids.append(row['__index_level_0__'])
        y = y.set_index(np.array(self._train_ids))
        df = X.add_column(self.taxa, lambda x : y[self.taxa])
        self._preprocessor = Chain(SimpleImputer(self.kmers, strategy = 'constant', fill_value = 0), MinMaxScaler(self.kmers))
        df = self._preprocessor.fit_transform(df)
        labels = np.unique(y[self.taxa])
        df = self._label_encode(df, labels)
        return df

    def _predict_preprocess(self, df):
        print('_predict_preprocess')
        for row in df.iter_rows():
            self._predict_ids.append(row['__index_level_0__'])
        self._preprocessor = Chain(SimpleImputer(self.kmers, strategy = 'constant', fill_value = 0), MinMaxScaler(self.kmers))
        df = self._preprocessor.fit_transform(df)
        return df

    @abstractmethod
    def _build(self):
        """
        """

    def train(self, X, y, cv = True):
        print('train')
        df = self._training_preprocess(X, y)
        if cv:
            self._cross_validation(df)
        else:
            datasets = {'train' : df}
            self._fit_model(datasets)

    @abstractmethod
    def _fit_model(self):
        """
        """

    def _cross_validation(self, df):
        print('_cross_validation')

        df_train, df_test = df.train_test_split(0.2, shuffle = True)

        datasets = {'train' : df_train, 'test' : df_test}
        self._fit_model(datasets)

        y_true = df_test.to_pandas()[self.taxa]
        df_test = df_test.drop_columns([self.taxa])
        y_pred = self.predict(df_test, cv = True)
        self._cv_score(y_true, y_pred)

    # Outputs scores for cross validation in a dictionnary
    def _cv_score(self, y_true, y_pred):
        print('_cv_score')

        if isinstance(y_pred, ray.data.dataset.Dataset):
            y_pred = y_pred.to_pandas()

        support = precision_recall_fscore_support(y_true, y_pred, average = 'weighted')

        scores = pd.DataFrame({'Classifier':self.classifier,'Precision':support[0],'Recall':support[1],'F-score':support[2]}, index = [1]).T

        scores.to_csv(self._cv_csv, header = False)

    @abstractmethod
    def predict(self):
        """
        """

    def _label_encode(self, df, labels):
        print('_label_encode')
        self._encoder = LabelEncoder(self.taxa)
        df = self._encoder.fit_transform(df)
        encoded = np.append(np.unique(df.to_pandas()[self.taxa]), -1)
        labels = np.append(labels, 'unknown')
        self._labels_map = zip(labels, encoded)
        return df

    def _label_decode(self, predict, threshold):
        print('_label_decode')
        predict = np.array(predict.to_pandas())
        decoded = pd.Series(np.empty(len(predict), dtype = object))
        for label, encoded in self._labels_map:
            decoded[predict == encoded] = label
        return decoded
