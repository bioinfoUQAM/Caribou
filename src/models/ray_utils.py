import os
import ray
import warnings
import pyarrow as pa
import pandas as pd

# Class construction
from abc import ABC, abstractmethod

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
    def __init__(
        self,
        classifier,
        dataset,
        outdir_model,
        outdir_results,
        batch_size,
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
        # Initialize empty
        self._labels_map = None
        self._predict_ids = []
        # Initialize Ray variables
        self._clf = None
        self._model_ckpt = None
        self._preprocessor = None
        self._encoder = None
        self._trainer = None
        self._train_params = {}
        self._predictor = None
        self._workdir = outdir_model
        # Files
        self._cv_csv = os.path.join(self.outdir_results,'{}_{}_K{}_cv_scores.csv'.format(self.classifier, self.taxa, self.k))

    @abstractmethod
    def _training_preprocess(self):
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

    def _sim_4_cv(self, df, kmers_ds, name):
        sim_genomes = []
        sim_taxas = []
        for row in df.iter_rows():
            sim_genomes.append(row['id'])
            sim_taxas.append(row[self.taxa])
        cls = pd.DataFrame({'id':sim_genomes,self.taxa:sim_taxas})
        sim_outdir = os.path.dirname(kmers_ds['profile'])
        cv_sim = readsSimulation(kmers_ds['fasta'], cls, sim_genomes, 'miseq', sim_outdir, name)
        sim_data = cv_sim.simulation(self.k, self.kmers)
        sim_ids = sim_data['ids']
        sim_ids = sim_data['ids']
        sim_cls = pd.DataFrame({'sim_id':sim_ids}, dtype = object)
        sim_cls['id'] = sim_cls['sim_id'].str.replace('_[0-9]+_[0-9]+_[0-9]+', '', regex=True)
        sim_cls = sim_cls.set_index('id').join(cls.set_index('id'))
        sim_cls = sim_cls.drop(['sim_id'], axis=1)
        sim_cls = sim_cls.reset_index(drop = True)
        df = ray.data.read_parquet(sim_data['profile'])
        df = self._scaler.transform(df)
        df = self._zip_X_y(df, sim_cls)
        return df

    def _cv_score(self, y_true, y_pred):
        print('_cv_score')

        support = precision_recall_fscore_support(y_true, y_pred, average = 'weighted')

        scores = pd.DataFrame({
            'Classifier':self.classifier,
            'Precision':support[0],
            'Recall':support[1],
            'F-score':support[2]
            },
            index = [1]
        ).T

        scores.to_csv(self._cv_csv, header = False)

    @abstractmethod
    def predict(self):
        """
        """

    @abstractmethod
    def _prob_2_cls(self):
        """
        """

    @abstractmethod
    def _label_encode(self):
        """
        """

    @abstractmethod
    def _label_decode(self):
        """
        """

    def _zip_X_y(self, X, y):
        num_blocks = int(X.num_blocks()/self.batch_size)
        len_x = X.count()
        self._ensure_length_ds(len_x,len(y))
        # Convert y -> ray.data.Dataset with arrow schema
        y = ray.data.from_arrow(pa.Table.from_pandas(y))
        # Repartition to 1 row/partition
        X = X.repartition(len_x)
        y = y.repartition(len_x)
        # Ensure both ds fully executed
        for ds in [X,y]:
            if not ds.is_fully_executed():
                ds.fully_executed()
        # Zip X and y
        df = X.zip(y).repartition(num_blocks)
# TODO: If still no work : write/read on disk + clear memory
        return df

    def _ensure_length_ds(self, len_x, len_y):
        if len_x != len_y:
            raise ValueError('X and y have different lengths: {} and {}'.format(len_x, len_y))