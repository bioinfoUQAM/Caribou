from .seq_collections import SeqCollection
from .kmer_collections import build_kmers_Xy_data, build_kmers

import os.path

import numpy as np
import pandas as pd

__author__ = "nicolas"

def build_load_save_data(seq_file, cls_file, prefix, dataset, k=4, full_kmers=False, low_var_threshold=None):

    # Generate the names of files
    Xy_file = prefix + "_K{}_Xy_genome_{}_data.npz".format(k,dataset)

    # Load if already exists
    if os.path.isfile(Xy_file) and save_Xy_data:
        data = load_Xy_data(Xy_file)

    else:
        # Build
        seq_data = SeqCollection((seq_file, cls_file))
        data = build_Xy_data(seq_data, k, full_kmers, low_var_threshold)
        # Save
        save_Xy_data(data, Xy_file)

    return data

# Build kmers collections
def build_Xy_data(seq_data, k, full_kmers=False, low_var_threshold=None):

    data = dict()

    X_train, y_train, kmers = build_kmers_Xy_data(seq_data, k,
        full_kmers = full_kmers,
        low_var_threshold = low_var_threshold,
        dtype = np.int32)

    X_test = "X_test"
    y_test = "y_test"

    # Data in a dictionnary
    data["X_train"] = X_train
    data["y_train"] = y_train
    data["X_test"] = X_test
    data["y_test"] = y_test
    data["kmers_list"] = kmers
    data["ids"] = seq_data.ids

    return data

# Load data from file to save on memory
def load_Xy_data(Xy_file):
    if os.path.basename(Xy_file).split(sep = ".")[1] == "hdf5":
        return pd.read_hdf(Xy_file, 'df')
    elif os.path.basename(Xy_file).split(sep = ".")[1] == "npz":
        with np.load(Xy_file, allow_pickle=True) as f:
            return f['data'].tolist()

# Save data to file to save on memory
def save_Xy_data(data,Xy_file):
    if type(data) == pd.core.frame.DataFrame:
        data.to_hdf(Xy_file, key='df', mode='w', complevel = 9, complib = 'bzip2')
    elif type(data) == dict:
        np.savez(Xy_file, data=data)
