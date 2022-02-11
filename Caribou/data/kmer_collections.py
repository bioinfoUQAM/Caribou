import os
import warnings

from subprocess import run
from shutil import rmtree

from joblib import Parallel, delayed, parallel_backend

from tensorflow.config import list_physical_devices

import numpy as np
import tables as tb
import pandas as pd
import dask.dataframe as dd

# From mlr_kgenomvir
__author__ = ['Amine Remita', 'Nicolas de Montigny']

__all__ = ['kmers_collection','construct_data','compute_seen_kmers_of_sequence','compute_given_kmers_of_sequence',
           'compute_kmers','threads','dask_client','build_kmers_Xy_data','build_kmers_X_data']

"""
Module adapted from module kmer_collections.py of
mlr_kgenomvir package [Remita et al. 2022]

Save kmers directly to drive instead of memory and
adapted / added functions to do so.
Converted to be only functions instead of object for parallelization.
"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# #####
# Data build functions
# ####################

def build_kmers_Xy_data(seq_data, k, Xy_file, length = 0, kmers_list = None):

    if kmers_list is not None:
        method = 'given'
    else:
        method = 'seen'

    collection = kmers_collection(seq_data, Xy_file, length, k, method = method, kmers_list = kmers_list)

    kmers_list = collection['kmers_list']
    X_data = collection['data']
    y_data = np.array(seq_data.labels)
    ids = seq_data.ids

    return X_data, y_data, kmers_list

def build_kmers_X_data(seq_data, X_file, kmers_list, k, length = 0):

    collection = kmers_collection(seq_data, X_file, length, k, method = 'given', kmers_list = kmers_list)
    kmers_list = collection['kmers_list']
    X_data = collection['data']
    ids = seq_data.ids

    return X_data, kmers_list, ids

# #####
# Kmers computing
# ##################

def kmers_collection(seq_data, Xy_file, length, k, method = 'seen', kmers_list = None):
    collection = {}
    #
    collection['data'] = Xy_file
    dir_path = os.path.split(Xy_file)[0] + "/tmp/"
    Xy_file = tb.open_file(Xy_file, "w")
    kmc_path = "{}/KMC/bin".format(os.path.dirname(os.path.realpath(__file__)))
    faSplit = "{}/faSplit".format(os.path.dirname(os.path.realpath(__file__)))
    #
    ddf = compute_kmers(seq_data, method, kmers_list, k, dir_path, faSplit, kmc_path)
    #
    collection['ids'], collection['kmers_list'] = construct_data(Xy_file, collection, ddf)
    #
    Xy_file.close()
    rmtree(dir_path)

    return collection

def construct_data(Xy_file, collection, ddf):
    # Extract ids and k-mers from dask dataframe
    ids = list(ddf.index)
    kmers_list = list(ddf.columns)

    # Convert dask df to numpy array and write directly to disk with pytables
    arr = np.array(ddf.to_dask_array(), dtype = np.int64)
    data = Xy_file.create_carray("/", "data", obj = arr)

    return ids, kmers_list

def compute_seen_kmers_of_sequence(kmc_path, k, dir_path, ind, file):
    # Make tmp folder per sequence
    tmp_folder = "{}tmp_{}/".format(dir_path, ind)
    os.mkdir(tmp_folder)
    # Count k-mers with KMC
    cmd_count = "{}/kmc -k{} -fm -cs1000000000 -m10 -hp {} {}/{} {}".format(kmc_path, k, file, tmp_folder, ind, tmp_folder)
    run(cmd_count, shell = True, capture_output=True)
    # Transform k-mers db with KMC
    cmd_transform = "{}/kmc_tools transform {}/{} dump {}/{}.txt".format(kmc_path, tmp_folder, ind, dir_path, ind)
    run(cmd_transform, shell = True, capture_output=True)
    # Parse k-mers file to dask dataframe
    id = os.path.splitext(os.path.basename(file))[0]
    profile = dd.from_pandas(pd.read_table('{}/{}.txt'.format(dir_path, ind), header = 0, names = [id], index_col = 0, dtype = object).T, chunksize = 1)

    return profile

def compute_given_kmers_of_sequence(kmers_list, kmc_path, k, dir_path, ind, file):
    # Make tmp folder per sequence
    tmp_folder = "{}tmp_{}".format(dir_path, ind)
    os.mkdir(tmp_folder)
    # Count k-mers with KMC
    cmd_count = "{}/kmc -k{} -fm -cs1000000000 -m10 -hp {} {}/{} {}".format(kmc_path, k, file, tmp_folder, ind, tmp_folder)
    run(cmd_count, shell = True, capture_output=True)
    # Transform k-mers db with KMC
    cmd_transform = "{}/kmc_tools transform {}/{} dump {}/{}.txt".format(kmc_path, tmp_folder, ind, dir_path, ind)
    run(cmd_transform, shell = True, capture_output=True)
    # Parse k-mers file to dask dataframe
    id = os.path.splitext(os.path.basename(file))[0]
    profile = pd.read_table('{}/{}.txt'.format(dir_path, ind), header = 0, names = [id], index_col = 0, dtype = object).T

    df = pd.DataFrame(np.zeros((1,len(kmers_list))), columns = kmers_list, index = [id])

    for kmer in kmers_list:
        if kmer in profile.columns:
            df.at[id,kmer] = profile.loc[id,kmer]
        else:
            df.at[id,kmer] = 0

    df = dd.from_pandas(df, chunksize = 1)
    return df

def compute_kmers(seq_data, method, kmers_list, k, dir_path, faSplit, kmc_path):
    file_list = []

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    cmd_split = '{} byname {} {}'.format(faSplit, seq_data.data, dir_path)

    os.system(cmd_split)

    for id in seq_data.ids:
        file = dir_path + id + '.fa'
        file_list.append(file)

    # Detect if a GPU is available
    if list_physical_devices('GPU'):
        ddf = parallel_GPU(file_list, method, kmers_list, kmc_path, k, dir_path)
    else:
        ddf = parallel_CPU(file_list, method, kmers_list, kmc_path, k, dir_path)

    return ddf

def parallel_CPU(file_list, method, kmers_list, kmc_path, k, dir_path):
    if method == 'seen':
        with parallel_backend('threading'):
            results = Parallel(n_jobs = -1, prefer = 'threads', verbose = 100)(
            delayed(compute_seen_kmers_of_sequence)
            (kmc_path, k, dir_path, i, file) for i, file in enumerate(file_list))
    elif method == 'given':
        with parallel_backend('threading'):
            results = Parallel(n_jobs = -1, prefer = 'threads', verbose = 100)(
            delayed(compute_given_kmers_of_sequence)
            (kmers_list, kmc_path, k, dir_path, i, file) for i, file in enumerate(file_list))

    for i in range(1, len(results)):
        if i == 1:
            ddf = dd.multi.concat([results[0], results[1]])
        else:
            ddf = ddf.append(results[i])

    return ddf

def parallel_GPU(file_list, method, kmers_list, kmc_path, k, dir_path):
    if method == 'seen':
        results = Parallel(n_jobs = -1, prefer = 'processes', verbose = 100)(
        delayed(compute_seen_kmers_of_sequence)
        (kmc_path, k, dir_path, i, file) for i, file in enumerate(file_list))
    elif method == 'given':
        results = Parallel(n_jobs = -1, prefer = 'processes', verbose = 100)(
        delayed(compute_given_kmers_of_sequence)
        (kmers_list, kmc_path, k, dir_path, i, file) for i, file in enumerate(file_list))

    for i in range(1, len(results)):
        if i == 1:
            ddf = dd.multi.concat([results[0], results[1]])
        else:
            ddf = ddf.append(results[i])

    return ddf
