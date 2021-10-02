from data.seq_collections import SeqCollection
from data.kmer_collections import build_kmers_Xy_data, build_kmers_X_data, build_kmers
from utils import load_Xy_data, save_Xy_data

import os.path

import numpy as np
import pandas as pd

__author__ = "nicolas"

def build_load_save_data(file, prefix, dataset, k=4, full_kmers=False, low_var_threshold=None):

    # Generate the names of files
    Xy_file = prefix + "_K{}_Xy_genome_{}_data.npz".format(k,dataset)


    # Load if already exists
    if os.path.isfile(Xy_file) and save_Xy_data:
        data = load_Xy_data(Xy_file)

    else:
        if isinstance(file, tuple):
            # Build
            print("seq_data")
            seq_data = SeqCollection((list(file)[0], list(file)[1]))
            print("Xy_data")
            data = build_Xy_data(seq_data, k, full_kmers, low_var_threshold)
            # Save
            save_Xy_data(data, Xy_file)
        else:
            # Build
            data = build_X_data(file, k, full_kmers, low_var_threshold)
            # Save
            save_Xy_data(data, Xy_file)
    return data

# Build kmers collections with known classes
def build_Xy_data(seq_data, k, full_kmers = False, low_var_threshold = None):
    data = dict()

    X, y, kmers = build_kmers_Xy_data(seq_data, k,
        full_kmers = full_kmers,
        low_var_threshold = low_var_threshold,
        dtype = np.int32)

    # Data in a dictionnary
    data["X"] = X
    data["y"] = y
    data["kmers_list"] = kmers
    data["ids"] = seq_data.ids

    return data

# Build kmers collection without known classes
def build_X_data(file, k, full_kmers = False, low_var_threshold = None):
    data = dict()

    X, kmers, ids = build_kmers_X_data(file, k,
        full_kmers = full_kmers,
        low_var_threshold = low_var_threshold,
        dtype = np.int32)

    # Data in a dictionnary
    data["X"] = X
    data["kmers_list"] = kmers
    data["ids"] = ids

    return data
