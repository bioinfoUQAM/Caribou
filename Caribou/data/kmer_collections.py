import os
import glob
import warnings

from subprocess import run
from shutil import rmtree

from joblib import Parallel, delayed, parallel_backend
from tensorflow.config import list_physical_devices

import numpy as np
import tables as tb
import pandas as pd

# Use cudf/dask_cudf only if GPU is available
if len(list_physical_devices('GPU')) > 0:
    import cudf
    import dask_cudf
    import dask.dataframe
    import dask.multiprocessing
    from dask.distributed import Client, wait
    from dask_cuda import LocalCUDACluster


# From mlr_kgenomvir
__author__ = ['Amine Remita', 'Nicolas de Montigny']

__all__ = ['kmers_collection','construct_data_GPU','construct_data_CPU','compute_seen_kmers_of_sequence','compute_given_kmers_of_sequence',
           'compute_kmers','parallel_CPU','parallel_GPU','build_kmers_Xy_data','build_kmers_X_data']

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
    kmc_path = "{}/KMC/bin".format(os.path.dirname(os.path.realpath(__file__)))
    faSplit = "{}/faSplit".format(os.path.dirname(os.path.realpath(__file__)))
    #
    collection['ids'], collection['kmers_list'] = compute_kmers(seq_data, method, kmers_list, k, dir_path, faSplit, kmc_path, Xy_file)
    #
    rmtree(dir_path)

    return collection

def construct_data_CPU(Xy_file, dir_path, list_id_file):
    df = None
    # Iterate over ids / files
    for id, file in list_id_file:
        if df is None:
            # Read first file to df directly
            df = pd.read_csv(file, sep = "\t", header = None, names = ['kmers', id], index_col=False)
            # Sort kmers column for faster join
            df = df.sort_values(by = 'kmers')
        else:
            # Read each file individually
            tmp = pd.read_csv(file, sep = "\t", header = None, names = ['kmers', id], index_col=False)
            # Sort kmers column for faster join
            tmp = tmp.sort_values(by = 'kmers')
            # Outer join each file to df
            df = df.merge(tmp, on = 'kmers', how = 'outer')

    # Extract ids and k-mers from df dataframe + remove kmers column
    kmers_list = list(df.loc[:, 'kmers'])
    df = df.drop(columns = 'kmers')
    ids = list(df.columns)

    # Convert pandas to numpy array and write directly to disk with pytables
    with tb.open_file(Xy_file, "w") as handle:
        data = handle.create_carray("/", "data", obj = np.array(df.T.fillna(0), dtype = np.int64))

    return ids, kmers_list

def construct_data_GPU(Xy_file, dir_path, list_id_file):
    ddf = None
    # Iterate over ids / files
    for id, file in list_id_file:
        if ddf is None:
            try:
                # Read first file to ddf directly
                ddf = dask_cudf.read_csv(file, sep = "\t", header = None, names = ['kmers', id])
                # Sort kmers column for faster join
                ddf = ddf.sort_values(by = 'kmers')
            except IndexError:
                print("Kmers extraction error for sequence {}".format(id))
        else:
            try:
                # Read each file individually
                tmp = dask_cudf.read_csv(file, sep = "\t", header = None, names = ['kmers', id])
                # Sort kmers column for faster join
                tmp = tmp.sort_values(by = 'kmers')
                # Outer join each file to ddf (fast according to doc)
                ddf = ddf.merge(tmp, on = 'kmers', how = 'outer')
            except IndexError:
                print("Kmers extraction error for sequence {}".format(id))

    # Extract ids and k-mers from dask_cudf dataframe + remove kmers column
    kmers_list = ddf.loc[:,'kmers'].compute().to_numpy()
    ddf = ddf.drop(columns = 'kmers')
    ids = list(ddf.columns)

    # Convert dask_cudf to numpy array and write directly to disk with pytables
    with tb.open_file(Xy_file, "w") as handle:
        data = handle.create_carray("/", "data", obj = ddf.fillna(0).compute().to_numpy().astype(np.int64).T)

    return ids, kmers_list

def compute_seen_kmers_of_sequence(kmc_path, k, dir_path, ind, file):
    if not os.path.isfile('{}/{}.csv'.format(dir_path, ind)):
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

        return id, "{}/{}.txt".format(dir_path, ind)

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
    # Temp pandas df to write given kmers to file
    df = pd.DataFrame(np.zeros((1,len(kmers_list))), columns = kmers_list, index = [id])

    for kmer in kmers_list:
        if kmer in profile.columns:
            df.at[id,kmer] = profile.loc[id,kmer]
        else:
            df.at[id,kmer] = 0

    df.T.to_csv('{}/{}.txt'.format(dir_path, ind), sep = "\t", header = False)

    return id, '{}/{}.txt'.format(dir_path, ind)

def compute_kmers(seq_data, method, kmers_list, k, dir_path, faSplit, kmc_path, Xy_file):
    file_list = []

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    cmd_split = '{} byname {} {}'.format(faSplit, seq_data.data, dir_path)

    os.system(cmd_split)

    for id in seq_data.ids:
        file = dir_path + id + '.fa'
        file_list.append(file)

    # Detect if a GPU is available
    if len(list_physical_devices('GPU')) > 0:
        list_id_file = parallel_GPU(file_list, method, kmers_list, kmc_path, k, dir_path)
        with LocalCUDACluster() as cluster, Client(cluster) as client:
            ids, kmers_list = construct_data_GPU(Xy_file, dir_path, list_id_file)
    else:
        list_id_file = parallel_CPU(file_list, method, kmers_list, kmc_path, k, dir_path)
        ids, kmers_list = construct_data_CPU(Xy_file, dir_path, list_id_file)

    return ids, kmers_list

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

    return results

def parallel_GPU(file_list, method, kmers_list, kmc_path, k, dir_path):
    if method == 'seen':
        results = Parallel(n_jobs = -1, prefer = 'processes', verbose = 100)(
        delayed(compute_seen_kmers_of_sequence)
        (kmc_path, k, dir_path, i, file) for i, file in enumerate(file_list))
    elif method == 'given':
        results = Parallel(n_jobs = -1, prefer = 'processes', verbose = 100)(
        delayed(compute_given_kmers_of_sequence)
        (kmers_list, kmc_path, k, dir_path, i, file) for i, file in enumerate(file_list))

    return results
