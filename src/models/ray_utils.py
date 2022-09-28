import os
import ray
import warnings
import numpy as np
import pandas as pd

from shutil import rmtree
from glob import glob

# Class construction
from abc import ABC, abstractmethod

# Data preprocessing
from ray.data.preprocessors import MinMaxScaler, Chain, SimpleImputer

# CV metrics
from sklearn.metrics import precision_recall_fscore_support

# Simulation class
from models.reads_simulation import readsSimulation

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
    def __init__(self, classifier, dataset, outdir_results, batch_size, k, taxa, kmers_list, verbose):
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
        # Initialize empty
        self._labels_map = None
        self._predict_ids = []
        # Initialize Ray variables
        self._n_workers = int(np.floor(os.cpu_count()*.8))
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

    @abstractmethod
    def _training_preprocess(self):
        """
        """

    def _predict_preprocess(self, df):
        print('_predict_preprocess')
        for row in df.iter_rows():
            self._predict_ids.append(row['__index_level_0__'])
        self._preprocessor = Chain(SimpleImputer(
            self.kmers, strategy='constant', fill_value=0), MinMaxScaler(self.kmers))
        df = self._preprocessor.fit_transform(df)
        return df

    def train(self, X, y, kmers_ds, cv = True):
        print('train')
        # cv = False
        df = self._training_preprocess(X, y)
        if cv:
            self._cross_validation(df, kmers_ds)
        else:
            df_train, df_val = df.drop_columns(['id']).train_test_split(0.2, shuffle = True)
            datasets = {'train' : ray.put(df_train), 'validation': ray.put(df_val)}
            self._fit_model(datasets)

    @abstractmethod
    def _fit_model(self):
        """
        """

    @abstractmethod
    def _cross_validation(self):
        """
        """

    def _sim_4_cv(self, df, kmers_ds, name):
        sim_genomes = []
        for row in df.iter_rows():
            sim_genomes.append(row['id'])
        cls = pd.DataFrame({'id':sim_genomes,self.taxa:df.to_pandas()[self.taxa]})
        sim_outdir = os.path.dirname(kmers_ds['profile'])
        cv_sim = readsSimulation(kmers_ds['fasta'], cls, sim_genomes, 'miseq', sim_outdir, name)
        sim_data = cv_sim.simulation(self.k, self.kmers)
        df = ray.data.read_parquet(sim_data['profile'])
        ids = []
        for row in df.iter_rows():
            ids.append(row['__index_level_0__'])
        labels = pd.DataFrame(sim_data['classes'], index = ids)
        df = df.add_column(self.taxa, lambda x : labels)
        return df

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

    @abstractmethod
    def _label_encode(self):
        """
        """

    def _label_decode(self, predict, threshold):
        print('_label_decode')
        predict = np.array(predict.to_pandas())
        decoded = pd.Series(np.empty(len(predict), dtype=object))
        for label, encoded in self._labels_map:
            decoded[predict == encoded] = label
        return decoded
