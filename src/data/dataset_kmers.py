from data.seq_collections import SeqCollection
from data.kmers_collection import KmersCollection

import os
import ray
import pickle

__author__ = 'Nicolas de Montigny'

__all__ = ['build_load_save_data', 'DatasetKmers', 'MergedDatabaseHostKmers', 'MetagenomeKmers']

class DatasetKmers():
    """
    A dataset to be used by Caribou from a database

    ----------
    Attributes
    ----------

    k : int
        The length of K-mers extracted

    dataset : string
        Name of the dataset from which the K-mers profiles were extracted

    profile : string
        Path to a folder containing a ray dataset of the K-mers extracted

    classes : list of strings
        List containing the known classes in the dataset

    taxas : list of strings
        List containing the known taxas for training

    kmers_list : list of strings
        List of K-mers extracted

    """
    def __init__(fasta_file, cls_file, prefix, dataset, k, kmers_list = None):
        # Variables
        self.k = k
        self.dataset = dataset
        # Hidden variables
        self._fasta_file = fasta_file
        self._cls_file = cls_file
        self._prefix = prefix
        # Initialize empty variables
        self.classes = []
        self.taxas = []
        self._seq_data = None
        self._kmers_collection = None
        # Conditional initialisation
        if isinstance(kmers_list, list):
            self.kmers_list = kmers_list
        else:
            self.kmers_list = None
        # Generate file names
        self._seqfile = os.path.join(self._prefix,'seqdata_{}.txt'.format(self.dataset))
        self.profile = os.path.join(self._prefix,'Xy_genome_{}_data_K{}'.format(self.dataset,self.k))
        self._data_file = os.path.join(self._prefix,'Xy_genome_{}_data_K{}.npz'.format(self.dataset,self.k))
        # Computation
        self._build_data()

    # Load file if already exists
    def _build_data(self):
        if os.path.isfile(self._data_file):
            _load_Xy_data(data_file)
        else:
            self._build_seq_data()
            self._build_kmers_collection()

    # Build Xy_data of database
    def _build_seq_data(self):
        if os.path.isfile(self._seqfile):
            with open(self._seqfile, 'rb') as handle:
                self._seq_data = pickle.load(handle)
        else:
            print('{} seq_data'.format(self.dataset))
            self._seq_data = SeqCollection((self._fasta_file, self._cls_file))
            with open(self._seqfile, 'wb') as handle:
                pickle.dump(self._seq_data, handle)

    # Build kmers collections with known classes and taxas
    def _build_kmers_collection(self):
        print('{} Xy_data, k = {}'.format(self.dataset, self.k))

        self._kmers_collection = KmersCollection(self._seq_data, self.profile, self.k, self.dataset, self.kmers_list)

        self.classes = self._kmers_collection.classes # Class labels
        self.kmers_list = self._kmers_collection.kmers_list # List of kmers
        self.taxas = self._kmers_collection.taxas # Known taxas for classification

        self._save_Xy_data()

    # Load data from file
    def _load_Xy_data(self):
        with np.load(self._data_file, allow_pickle=True) as f:
            data = f['data'].tolist()
        self.profile = data['profile']
        self.classes = data['classes']
        self.kmers_list = data['kmers']
        self.taxas = data['taxas']

    # Save data to file
    def _save_Xy_data(self):
        data = {}
        data['profile'] = self.profile
        data['classes'] = self.classes
        data['kmers'] = self.kmers_list
        data['taxas'] = self.taxas
        np.savez(self._data_file, data = data)


class MergedDatabaseHostKmers(DatasetKmers):
    """
    Merged dataset of Bacterial and Host databases

    Attributes are the same as DatasetKmers
    """
    def __init__(db_dataset, host_dataset):
        # Variables
        self.k = db_dataset.k
        self.taxas = db_dataset.taxas
        self.host = host_dataset.dataset
        self.dataset = db_dataset.dataset
        self.kmers_list = db_dataset.kmers_list
        self.profile = os.path.join(os.path.splitext(database_data.profile)[0], "_host_merged")
        self.classes = np.array(pd.DataFrame(db_dataset.classes, columns = db_dataset.taxas).append(pd.DataFrame(host_dataset.classes, columns = host_dataset.taxas), ignore_index = True))
        # Computation
        self._merge_database_host(db_dataset.profile, host_dataset.profile)

    def _merge_database_host(self, db_profile, host_profile):

        df_db = ray.data.read_parquet(db_profile)
        df_host = ray.data.read_parquet(host_profile)
        df_merged = df_db.union(df_host)
        df_merged.write_parquet(self.profile)

        self._save_Xy_data()

class MetagenomeKmers(DatasetKmers):
    """
    A dataset of a sequenced metagenome to be classified by Caribou

    ----------
    Attributes
    ----------

    k : int
        The length of K-mers extracted

    dataset : string
        Name of the dataset from which the K-mers profiles were extracted

    profile : string
        Path to a folder containing a ray dataset of the K-mers extracted

    kmers_list : list of strings
        List of K-mers extracted

    """
    def __init__(fasta_file, prefix, dataset, host, kmers_list, k):
        # Variables
        self.k = k
        self.dataset = dataset
        self.kmers_list = kmers_list
        # Hidden variables
        self._prefix = prefix
        self._fasta_file = fasta_file
        # Initialize empty variables
        self._seq_data = None
        self._kmers_collection = None
        # Generate file names
        self._seqfile = os.path.join(self._prefix,'seqdata_{}.txt'.format(self.dataset))
        self.profile = os.path.join(self._prefix,'Xy_genome_{}_data_K{}'.format(self.dataset,self.k))
        self._data_file = os.path.join(self._prefix,'Xy_genome_{}_data_K{}.npz'.format(self.dataset,self.k))
        # Computation
        self._build_data()

    # Build kmers collection with unknown classes
    def _build_kmers_collection(self):
        print('{} X_data, k = {}'.format(self.dataset, self.k))

        self._kmers_collection = KmersCollection(self._seq_data, self.profile, self.k, self.dataset, self.kmers_list)

        self._save_Xy_data()

    # Load data from file
    def _load_Xy_data(self):
        with np.load(self._data_file, allow_pickle=True) as f:
            data = f['data'].tolist()
        self.profile = data['profile']
        self.kmers_list = data['kmers']

    # Save data to file
    def _save_Xy_data(self):
        data = {}
        data['profile'] = self.profile
        data['kmers'] = self.kmers_list
        np.savez(self._data_file, data = data)

################################################################################
# Caller function
################################################################################
def build_load_save_data(file, hostfile, prefix, dataset, host, kmers_list=None, k=4):
    dataset = None
    db_dataset = None
    host_dataset = None
    if isinstance(file, tuple) and isinstance(hostfile, tuple):
        db_dataset = DatasetKmers(file[0], file[1], prefix, dataset, k)
        host_dataset = DatasetKmers(hostfile[0], hostfile[1], prefix, host, k, db_dataset.kmers_list)
        dataset = MergedDatabaseHostKmers(db_dataset, host_dataset)
    elif isinstance(file, tuple) and not isinstance(hostfile, tuple):
        dataset = DatasetKmers(file[0], file[1], prefix, dataset, k)
    elif not isinstance(file, tuple) and isinstance(hostfile, tuple):
        dataset = DatasetKmers(hostfile[0], hostfile[1], prefix, host, k, kmers_list)
    else:
        dataset = MetagenomeKmers(file, prefix, k, kmers_list)

    return dataset
