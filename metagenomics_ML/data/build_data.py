from data.seq_collections import SeqCollection
from data.kmer_collections import build_kmers_Xy_data, build_kmers_X_data, build_kmers
from utils import load_Xy_data, save_Xy_data

import os.path

import numpy as np
import pandas as pd

import pickle

__author__ = "nicolas"

def build_load_save_data(file, hostfile, prefix, dataset, k=4, full_kmers=False, low_var_threshold=None):

    # Generate the names of files
    Xy_file = prefix + "_K{}_Xy_genome_{}_data.h5f".format(k,dataset)
    data_file = prefix + "_K{}_Xy_genome_{}_data.npz".format(k,dataset)

    # Load file if already exists
    if os.path.isfile(data_file):
        data = load_Xy_data(data_file)

    else:
        if isinstance(file, tuple):
# To accelerate testing with bigger database
            if not os.path.isfile("/home/nicolas/github/metagenomics_ML/data/output/mock/seq_data.txt"):
                print("seq_data")
                seq_data = SeqCollection((list(file)[0], list(file)[1]))
                with open("/home/nicolas/github/metagenomics_ML/data/output/mock/seq_data.txt", "wb") as handle:
                    pickle.dump(seq_data, handle)
                if isinstance(hostfile, tuple):
                    seq_data_host = SeqCollection((list(hostfile)[0], list(hostfile)[1]))
                    with open("/home/nicolas/github/metagenomics_ML/data/output/mock/seq_data_host.txt", "wb") as handle:
                        pickle.dump(seq_data_host, handle)
            else:
                with open("/home/nicolas/github/metagenomics_ML/data/output/mock/seq_data.txt", "rb") as handle:
                    seq_data = pickle.load(handle)
                if isinstance(hostfile, tuple):
                    with open("/home/nicolas/github/metagenomics_ML/data/output/mock/seq_data_host.txt", "rb") as handle:
                        seq_data_host = pickle.load(handle)

            # Build Xy_data to drive
            if isinstance(hostfile, tuple):
                print("Xy_data with host")
                data = build_Xy_data((seq_data, seq_data_host), k, Xy_file, seq_data.length, full_kmers, low_var_threshold)
            else:
                print("Xy_data without host")
                data = build_Xy_data(seq_data, k, Xy_file, seq_data.length, full_kmers, low_var_threshold)

            save_Xy_data(data, data_file)

        else:
            # Build X_data to drive
            print("X_data")
            seq_data = SeqCollection(file)
            data = build_X_data(seq_data, k, Xy_file, seq_data.length, full_kmers, low_var_threshold)
            save_Xy_data(data, data_file)
    return data

# Build kmers collections with known classes
def build_Xy_data(seq_data, k, Xy_file, length = 0, full_kmers = False, low_var_threshold = None):
    data = dict()

    X, y, kmers = build_kmers_Xy_data(seq_data, k, Xy_file,
        length = length,
        full_kmers = full_kmers,
        low_var_threshold = low_var_threshold,
        dtype = np.float32)

    # Data in a dictionnary
    data["X"] = str(Xy_file)
    data["y"] = y
    data["kmers_list"] = kmers
    if seq_data[1] == "none":
        data["ids"] = seq_data[0].ids
    else:
        data["ids"] = np.append(seq_data[0].ids, seq_data[1].ids)

    return data

# Build kmers collection without known classes
def build_X_data(seq_data, k, X_file, length = 0, full_kmers = False, low_var_threshold = None):
    data = dict()

    X, kmers, ids = build_kmers_X_data(seq_data, k, X_file,
        length = length,
        full_kmers = full_kmers,
        low_var_threshold = low_var_threshold,
        dtype = np.float32)

    # Data in a dictionnary
    data["X"] = str(X_file)
    data["kmers_list"] = kmers
    data["ids"] = ids

    return data
