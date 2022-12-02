import os
import ray
import warnings
import pandas as pd
import pyarrow as pa

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
        df = ray.data.read_parquet(sim_data['profile'])
        labels = ray.data.from_arrow(
            pa.Table.from_pandas(
                pd.DataFrame(
                    sim_data['classes'],
                    columns = [self.taxa])
                )).repartition(
                    df.num_blocks())
        df = df.repartition(df.num_blocks()).zip(labels)
        return df

    def _cv_score(self, y_true, y_pred):
        print('_cv_score')

        support = precision_recall_fscore_support(y_true, y_pred, average = 'weighted')

        scores = pd.DataFrame({'Classifier':self.classifier,'Precision':support[0],'Recall':support[1],'F-score':support[2]}, index = [1]).T

        scores.to_csv(self._cv_csv, header = False)

    @abstractmethod
    def predict(self):
        """
        """

    def _prob_2_cls(self, predict, nb_cls, threshold):
        print('_prob_2_cls')
        if nb_cls == 1 and self.classifier != 'lstm':
            predict = np.round(abs(np.concatenate(predict.to_pandas()['predictions'])))
        else:
            predict = predict.map_batches(map_predicted_label, threshold)
            predict = np.ravel(np.array(predict.to_pandas()))
        return predict

    @abstractmethod
    def _label_encode(self):
        """
        """

    @abstractmethod
    def _label_decode(self):
        """
        """

# Mapping function outside of the class as mentioned on the Ray discussion
# https://discuss.ray.io/t/statuscode-resource-exhausted/4379/16
################################################################################

def map_predicted_label(df, threshold):
    predict = pd.DataFrame({'best_proba': [df['predictions'][i][np.argmax(df['predictions'][i])] for i in range(len(df))],
                            'predicted_label': [np.argmax(df['predictions'][i]) for i in range(len(df))]})
    predict.loc[predict['best_proba'] < threshold, 'predicted_label'] = -1
    return pd.DataFrame(predict['predicted_label'])
