
from utils import load_Xy_data, save_Xy_data
from data.seq_collections import SeqCollection
from data.kmers_collection import KmersCollection

import os
import ray
import pickle

__author__ = 'Nicolas de Montigny'

__all__ = ['build_load_save_data', 'build_Xy_data', 'build_X_data']

class DatasetKmers():
    def __init__(file, prefix, dataset, k, kmers_list = None):
        # Variables
        self.k = k
        self.dataset = dataset
        # Hidden variables
        self._file = file
        self._prefix = prefix
        # Initialize empty variables
        self.data = {}
        # Conditional initialisation
        if isinstance(kmers_list, list):
            self.kmers_list = kmers_list
        else:
            self.kmers_list = []
        # Generate file names
        self._seqfile = os.path.join(prefix,'seqdata_{}.txt'.format(dataset))
        self._Xy_file = os.path.join(prefix,'Xy_genome_{}_data_K{}'.format(dataset,k))
        self._data_file = os.path.join(prefix,'Xy_genome_{}_data_K{}.npz'.format(dataset,k))

    # Load file if already exists
    def _build_data(self):
        if os.path.isfile(self._data_file):
            self.data = load_Xy_data(data_file)
        else:
            self._build_seq_data()
            self._build_kmers_collection()

    # Build Xy_data of database
    def _build_seq_data(self):
        if os.path.isfile(self.seqfile):
        else:
            print('Database seq_data')
            seq_data = SeqCollection((list(file)[0], list(file)[1]))
            with open(seqfile, 'wb') as handle:
                pickle.dump(seq_data, handle)

    # Build Xy_data to drive
    def _build_kmers_collection(self):
        print('Database Xy_data, k = {}'.format(self.k))
        self._build_Xy_data()
        self._save_Xy_data()

    # Load data from file
    def load_Xy_data(self):
        with np.load(self._Xy_file, allow_pickle=True) as f:
            self.data = f['data'].tolist()

    # Save data to file
    def save_Xy_data(self):
        np.savez(self._Xy_file, data = self.data)


class MergedDatabaseHostKmers(DatasetKmers):
    def __init__(db_dataset, host_dataset):

# TODO: REARRANGE FOR CLASS
    def merge_database_host(database_data, host_data):
        merged_data = {}

        merged_file = "{}_host_merged".format(os.path.splitext(database_data["profile"])[0])

        merged_data['profile'] = merged_file # Kmers profile
        merged_data['classes'] = np.array(pd.DataFrame(database_data["classes"], columns = database_data["taxas"]).append(pd.DataFrame(host_data["classes"], columns = host_data["taxas"]), ignore_index = True)) # Class labels
        merged_data['kmers'] = database_data["kmers"] # Features
        merged_data['taxas'] = database_data["taxas"] # Known taxas for classification

        df_db = ray.data.read_parquet(database_data["profile"])
        df_host = ray.data.read_parquet(host_data["profile"])
        df_merged = df_db.union(df_host)
        df_merged.write_parquet(merged_file)

        return merged_data

class MetagenomeKmers(DatasetKmers):
    def __init__(file, prefix, dataset, host, kmers_list, k):
        super().__init__(file, prefix, dataset, k):

# Caller function
################################################################################
def build_load_save_data(file, hostfile, prefix, dataset, host, kmers_list=None, k=4):
    dataset = None
    if isinstance(file, tuple) and isinstance(hostfile, tuple):
        db_dataset = DatasetKmers(file, prefix, dataset, k)
        host_dataset = DatasetKmers(hostfile, prefix, host, k, db_dataset.kmers_list)
        dataset = MergedDatabaseHostKmers(db_dataset, host_dataset)
    elif isinstance(file, tuple) and not isinstance(hostfile, tuple):
        dataset = DatasetKmers(file, prefix, dataset, k)
    elif not isinstance(file, tuple) and isinstance(hostfile, tuple):
        dataset = DatasetKmers(hostfile, prefix, host, k, kmers_list)
    else:
        dataset = MetagenomeKmers(hostfile, prefix, host, k, kmers_list)

    return dataset



        # Assign kmers_list to variable ater extracting database data
        if kmers_list is None:
            kmers_list = data['kmers']

        # Build Xy_data of host
        if isinstance(hostfile, tuple) and kmers_list is not None:
            if not os.path.isfile(seqfile_host):
                print('Host seq_data')
                seq_data_host = SeqCollection((list(hostfile)[0], list(hostfile)[1]))
                with open(seqfile_host, 'wb') as handle:
                    pickle.dump(seq_data_host, handle)
            else:
                with open(seqfile_host, 'rb') as handle:
                    seq_data_host = pickle.load(handle)

            # Build Xy_data to drive
            print('Host Xy_data, k = {}'.format(k))
            data_host = build_Xy_data(seq_data_host, k, Xy_file_host, dataset, kmers_list)
            save_Xy_data(data_host, data_file_host)

        # Build X_data of dataset to analyse
        if not isinstance(file, tuple) and not isinstance(hostfile, tuple) and kmers_list is not None:
            print('Dataset seq_data')
            seq_data = SeqCollection(file)
            print('Dataset X_data, k = {}'.format(k))
            data = build_X_data(seq_data, k, Xy_file, dataset, kmers_list)
            save_Xy_data(data, data_file)

    if data is not None and data_host is None:
        return data
    elif data is None and data_host is not None:
        return data_host
    else:
        return data, data_host

# Build kmers collections with known classes and taxas
def build_Xy_data(seq_data, k, Xy_file, dataset, kmers_list = None):
    data = {}

    collection = KmersCollection(seq_data, Xy_file, k, dataset, kmers_list)

    # Data in a dictionnary
    data['profile'] = collection.Xy_file # Kmers profile
    data['classes'] = collection.classes # Class labels
    data['kmers'] = collection.kmers_list # Features
    data['taxas'] = collection.taxas # Known taxas for classification

    return data

# Build kmers collection with unknown classes
def build_X_data(seq_data, k, X_file, dataset, kmers_list):
    data = {}

    collection = KmersCollection(seq_data, X_file, k, dataset, kmers_list)

    # Data in a dictionnary
    data['profile'] = collection.Xy_file
    data['kmers'] = collection.kmers

    return data
