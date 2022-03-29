from Caribou.data.seq_collections import SeqCollection
from Caribou.data.kmer_collections import build_kmers_Xy_data, build_kmers_X_data
from Caribou.utils import load_Xy_data, save_Xy_data

import os.path

import numpy as np
import pandas as pd

import pickle

__author__ = "Nicolas de Montigny"

__all__ = ['build_load_save_data', 'build_Xy_data', 'build_X_data']

def build_load_save_data(file, hostfile, prefix, dataset, host, kmers_list=None, k=4):
    # Declare data variables as none
    data = None
    data_host = None

    # Generate the names of files
    Xy_file = "{}/Xy_genome_{}_data_K{}.h5f".format(prefix,dataset,k)
    data_file = "{}/Xy_genome_{}_data_K{}.npz".format(prefix,dataset,k)
    Xy_file_host = "{}/Xy_genome_{}_data_K{}.h5f".format(prefix,host,k)
    data_file_host = "{}/Xy_genome_{}_data_K{}.npz".format(prefix,host,k)
    seqfile = "{}/seqdata_{}.txt".format(prefix, dataset)
    seqfile_host = "{}/seqdata_{}.txt".format(prefix, dataset)

    # Load file if already exists
    if os.path.isfile(data_file) and os.path.isfile(data_file_host) and isinstance(hostfile, tuple):
        data = load_Xy_data(data_file)
        data_host = load_Xy_data(data_file_host)
    elif os.path.isfile(data_file):
        data = load_Xy_data(data_file)
    else:
        # Build Xy_data of database
        if isinstance(file, tuple):
            if not os.path.isfile(seqfile):
                print("Database seq_data")
                seq_data = SeqCollection((list(file)[0], list(file)[1]))
                with open(seqfile, "wb") as handle:
                    pickle.dump(seq_data, handle)
            else:
                with open(seqfile, "rb") as handle:
                    seq_data = pickle.load(handle)

            # Build Xy_data to drive
            print("Database Xy_data, k = {}".format(k))
            data = build_Xy_data(seq_data, k, Xy_file, dataset, seq_data.length, kmers_list = None)
            save_Xy_data(data, data_file)

        # Assing kmers_list to variable ater extracting database data
        if kmers_list is None and isinstance(data['kmers_list'], list):
            kmers_list = data['kmers_list']

        # Build Xy_data of host
        if isinstance(hostfile, tuple) and kmers_list is not None:
            if not os.path.isfile(seqfile_host):
                print("Host seq_data")
                seq_data_host = SeqCollection((list(hostfile)[0], list(hostfile)[1]))
                with open(seqfile_host, "wb") as handle:
                    pickle.dump(seq_data_host, handle)
            else:
                with open(seqfile_host, "rb") as handle:
                    seq_data_host = pickle.load(handle)

            # Build Xy_data to drive
            print("Host Xy_data, k = {}".format(k))
            data_host = build_Xy_data(seq_data_host, k, Xy_file_host, dataset, seq_data_host.length, kmers_list)
            save_Xy_data(data_host, data_file_host)

        # Build X_data of dataset to analyse
        if not isinstance(file, tuple) and not isinstance(hostfile, tuple) and kmers_list is not None:
            print("Dataset seq_data")
            seq_data = SeqCollection(file)
            print("Dataset X_data, k = {}".format(k))
            data = build_X_data(seq_data, k, Xy_file, kmers_list, dataset, seq_data.length)
            save_Xy_data(data, data_file)

    if data is not None and data_host is None:
        return data
    elif data is None and data_host is not None:
        return data_host
    else:
        return data, data_host

# Build kmers collections with known classes
def build_Xy_data(seq_data, k, Xy_file, dataset, length = 0, kmers_list = None):
    data = dict()

    X, y, kmers = build_kmers_Xy_data(seq_data, k, Xy_file,
                                      dataset,
                                      length = length,
                                      kmers_list = kmers_list)

    # Data in a dictionnary
    data["X"] = str(Xy_file)
    data["y"] = y
    data["kmers_list"] = kmers
    data["ids"] = seq_data.ids
    data["taxas"] = seq_data.taxas

    return data

# Build kmers collection without known classes
def build_X_data(seq_data, k, X_file, kmers_list, dataset, length = 0):
    data = dict()

    X, kmers, ids = build_kmers_X_data(seq_data, X_file,
                                       k,
                                       dataset,
                                       kmers_list = kmers_list,
                                       length = length)

    # Data in a dictionnary
    data["X"] = str(X_file)
    data["kmers_list"] = kmers
    data["ids"] = ids
    data["taxas"] = seq_data.taxas

    return data
