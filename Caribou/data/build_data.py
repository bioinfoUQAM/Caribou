from Caribou.data.seq_collections import SeqCollection
from Caribou.data.kmer_collections import build_kmers_Xy_data, build_kmers_X_data, build_kmers
from Caribou.utils import load_Xy_data, save_Xy_data

import os.path

import numpy as np
import pandas as pd

import pickle

__author__ = "Nicolas de Montigny"

__all__ = ['build_load_save_data', 'build_Xy_data', 'build_X_data']

def build_load_save_data(file, hostfile, prefix, dataset, kmers_list=None, k=4, full_kmers=False, low_var_threshold=None):

    # Generate the names of files
    Xy_file = "{}_K{}_Xy_genome_{}_data.h5f".format(prefix,k,dataset)
    data_file = "{}_K{}_Xy_genome_{}_data.npz".format(prefix,k,dataset)
    Xy_file_host = "{}_K{}_Xy_genome_{}_host_data.h5f".format(prefix,k,dataset)
    data_file_host = "{}_K{}_Xy_genome_{}_host_data.npz".format(prefix,k,dataset)
    seqfile = "{}_seqdata_db_{}.txt".format(prefix, dataset)
    seqfile_host = "{}_seqdata_host_{}.txt".format(prefix, dataset)

    # Load file if already exists
    if os.path.isfile(data_file) and os.path.isfile(data_file_host) and isinstance(hostfile, tuple):
        data = load_Xy_data(data_file)
        data_host = load_Xy_data(data_file_host)
        return data, data_host
    elif os.path.isfile(data_file):
        data = load_Xy_data(data_file)
        return data
    else:
        # Build Xy_data of database and host
        if isinstance(file, tuple):
            if not os.path.isfile(seqfile):
                print("seq_data")
                seq_data = SeqCollection((list(file)[0], list(file)[1]))
                with open(seqfile, "wb") as handle:
                    pickle.dump(seq_data, handle)
                if isinstance(hostfile, tuple):
                    seq_data_host = SeqCollection((list(hostfile)[0], list(hostfile)[1]))
                    with open(seqfile_host, "wb") as handle:
                        pickle.dump(seq_data_host, handle)
            else:
                with open(seqfile, "rb") as handle:
                    seq_data = pickle.load(handle)
                if isinstance(hostfile, tuple):
                    with open(seqfile_host, "rb") as handle:
                        seq_data_host = pickle.load(handle)

            # Build Xy_data to drive
            if isinstance(hostfile, tuple):
                print("Xy_data with host, k = {}".format(k))
                data = build_Xy_data(seq_data, k, Xy_file, seq_data.length, kmers_list = None, full_kmers = full_kmers, low_var_threshold = low_var_threshold)
                save_Xy_data(data, data_file)
                data_host = build_Xy_data(seq_data_host, k, Xy_file_host, seq_data_host.length, kmers_list = data["kmers_list"], full_kmers = full_kmers, low_var_threshold = low_var_threshold)
                save_Xy_data(data_host, data_file_host)
                return data, data_host
            else:
                print("Xy_data without host, k = {}".format(k))
                data = build_Xy_data(seq_data, k, Xy_file, seq_data.length, full_kmers, low_var_threshold)
                save_Xy_data(data, data_file)
                return data

        else:
            # Build X_data of dataset
            print("X_data, k = {}".format(k))
            seq_data = SeqCollection(file)
            data = build_X_data(seq_data, Xy_file, kmers_list, seq_data.length)
            save_Xy_data(data, data_file)
            return data

# Build kmers collections with known classes
def build_Xy_data(seq_data, k, Xy_file, length = 0, kmers_list = None, full_kmers = False, low_var_threshold = None):
    data = dict()

    X, y, kmers = build_kmers_Xy_data(seq_data, k, Xy_file,
        length = length,
        full_kmers = full_kmers,
        kmers_list = kmers_list,
        low_var_threshold = low_var_threshold,
        dtype = np.float32)

    # Data in a dictionnary
    data["X"] = str(Xy_file)
    data["y"] = y
    data["kmers_list"] = kmers
    data["ids"] = seq_data.ids
    data["taxas"] = seq_data.taxas

    return data

# Build kmers collection without known classes
def build_X_data(seq_data, X_file, kmers_list, length = 0):
    data = dict()

    X, kmers, ids = build_kmers_X_data(seq_data, X_file, kmers_list, length = length, dtype = np.float32)

    # Data in a dictionnary
    data["X"] = str(X_file)
    data["kmers_list"] = kmers
    data["ids"] = ids
    data["taxas"] = seq_data.taxas

    return data
