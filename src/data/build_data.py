
import os
import pickle
import numpy as np

from utils import load_Xy_data, save_Xy_data
from data.seq_collections import SeqCollection
from data.kmers_collection import KmersCollection

__author__ = 'Nicolas de Montigny'

__all__ = ['build_load_save_data', 'build_Xy_data', 'build_X_data']


def build_load_save_data(file, hostfile, prefix, dataset, host, kmers_list=None, k=20, features_threshold = np.inf, nb_features_keep = np.inf):
    # Declare data variables as none
    data = None
    data_host = None
    # Generate the names of files
    Xy_file = os.path.join(prefix, 'Xy_genome_{}_data_K{}'.format(dataset, k))
    data_file = os.path.join(prefix, 'Xy_genome_{}_data_K{}.npz'.format(dataset, k))
    Xy_file_host = os.path.join(prefix, 'Xy_genome_{}_data_K{}'.format(host, k))
    data_file_host = os.path.join(prefix, 'Xy_genome_{}_data_K{}.npz'.format(host, k))
    seqfile = os.path.join(prefix, 'seqdata_{}.txt'.format(dataset))
    seqfile_host = os.path.join(prefix, 'seqdata_{}.txt'.format(host))

    # Load file if already exists
    if os.path.isfile(data_file) and os.path.isfile(data_file_host) and isinstance(hostfile, tuple):
        data = load_Xy_data(data_file)
        data_host = load_Xy_data(data_file_host)
    elif os.path.isfile(data_file) :
        data = load_Xy_data(data_file)
    else:
        # Build Xy_data of database
        if isinstance(file, tuple):
            if not os.path.isfile(seqfile):
                print('Database seq_data')
                seq_data = SeqCollection((list(file)[0], list(file)[1]))
                with open(seqfile, 'wb') as handle:
                    pickle.dump(seq_data, handle)
            else:
                with open(seqfile, 'rb') as handle:
                    seq_data = pickle.load(handle)

            # Build Xy_data to drive
            print('Database Xy_data, k = {}'.format(k))
            data = build_Xy_data(
                seq_data,
                k,
                Xy_file,dataset,
                kmers_list=None,
                features_threshold = features_threshold,
                nb_features_keep = nb_features_keep)
            save_Xy_data(data, data_file)

        # Assign kmers_list to variable after extracting database data
        if kmers_list is None:
            kmers_list = data['kmers']

        # Build Xy_data of host
        if isinstance(hostfile, tuple) and kmers_list is not None:
            if not os.path.isfile(seqfile_host):
                print('Host/simulated seq_data')
                seq_data_host = SeqCollection((list(hostfile)[0], list(hostfile)[1]))
                with open(seqfile_host, 'wb') as handle:
                    pickle.dump(seq_data_host, handle)
            else:
                with open(seqfile_host, 'rb') as handle:
                    seq_data_host = pickle.load(handle)

            # Build Xy_data to drive
            print('Host/simulated Xy_data, k = {}'.format(k))
            data_host = build_Xy_data(
                seq_data_host,
                k,
                Xy_file_host,
                dataset,
                kmers_list)
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
def build_Xy_data(seq_data, k, Xy_file, dataset, kmers_list=None, features_threshold = np.inf, nb_features_keep = np.inf):
    data = {}

    collection = KmersCollection(
        seq_data,
        Xy_file,
        k,
        dataset,
        kmers_list,
        features_threshold,
        nb_features_keep)

    # Data in a dictionnary
    data['profile'] = collection.Xy_file  # Kmers profile
    data['ids'] = collection.ids  # Ids of profiles
    data['classes'] = collection.classes  # Class labels
    data['kmers'] = collection.kmers_list  # Features
    data['taxas'] = collection.taxas  # Known taxas for classification
    data['fasta'] = collection.fasta  # Fasta file -> simulate reads if cv

    return data

# Build kmers collection with unknown classes
def build_X_data(seq_data, k, X_file, dataset, kmers_list):
    data = {}

    collection = KmersCollection(
        seq_data,
        X_file,
        k,
        dataset,
        kmers_list)

    # Data in a dictionnary
    data['profile'] = collection.Xy_file
    data['ids'] = collection.ids
    data['kmers'] = collection.kmers_list

    return data
