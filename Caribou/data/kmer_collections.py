import os
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
    from dask.distributed import Client, wait
    from dask_cuda import LocalCUDACluster
else:
    from collections import defaultdict


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

def construct_data_CPU(Xy_file, results):
    dict_data = defaultdict(lambda: [0]*len(results))
    ids = []
    for ind, result in enumerate(results):
        kmers_list = list(result)
        ids.append(result.index[0])
        for kmer in kmers_list:
            dict_data[kmer][ind] = result[kmer][0]

    print('kmers_list :', kmers_list)
    print('ids : ', ids)

    with tb.open_file(Xy_file, "w") as handle:
        data = handle.create_carray("/", "data", obj = np.array([ dict_data[x] for x in dict_data ], dtype=np.int64).T)

    return ids, kmers_list

def construct_data_GPU(Xy_file, dir_path):
    # List files in directory
    file_list = os.listdir(dir_path)
    # Append each row to the dask_cuDF
    for i, file in enumerate(file_list):
        if i == 0:
            ddf = dask_cudf.read_csv(file_list[0])
        else:
            tmp_df = dask_cudf.read_csv(file_list[i])
            print(tmp_df)
            ddf = ddf.merge(tmp_df, on = 'index', how = 'left')
    # Dask_cudf read all .txt in folder and concatenate
    #ddf = dask_cudf.read_csv('{}/*.csv'.format(dir_path), index = [id])
    # Extract ids and k-mers from dask dataframe
    ids = list(ddf.index.compute())
    kmers_list = len(list(ddf.columns.compute()))

    print('kmers_list :', kmers_list)
    print('ids : ', ids)

    # Convert dask df to numpy array and write directly to disk with pytables
    arr = ddf.compute().as_matrix()
    wait(arr)
    with tb.open_file(Xy_file, "w") as handle:
        data = handle.create_carray("/", "data", obj = arr)

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
    df = pd.read_table('{}/{}.txt'.format(dir_path, ind), header = 0, names = [id], index_col = 0, dtype = object).T
    df.to_csv('{}/{}.csv'.format(dir_path, ind))
    #df_file = '{}/{}.txt'.format(dir_path, ind)

    #return df_file

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

    return df

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
        parallel_GPU(file_list, method, kmers_list, kmc_path, k, dir_path)
        with LocalCUDACluster() as cluster, Client(cluster) as client:
            ids, kmers_list = construct_data_GPU(Xy_file, dir_path)
    else:
        results = parallel_CPU(file_list, method, kmers_list, kmc_path, k, dir_path)
        ids, kmers_list = construct_data_CPU(Xy_file, results)

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
        Parallel(n_jobs = -1, prefer = 'processes', verbose = 100)(
        delayed(compute_seen_kmers_of_sequence)
        (kmc_path, k, dir_path, i, file) for i, file in enumerate(file_list))
    elif method == 'given':
        Parallel(n_jobs = -1, prefer = 'processes', verbose = 100)(
        delayed(compute_given_kmers_of_sequence)
        (kmers_list, kmc_path, k, dir_path, i, file) for i, file in enumerate(file_list))

    """
            df = dask_cudf.from_cudf(cudf.read_csv(, sep = "\t", header = 0, names = [id], index_col = 0, dtype = object).T, chunksize = 1)
        else:
            df = pd.read_table('{}/{}.txt'.format(dir_path, ind), header = 0, names = [id], index_col = 0, dtype = object).T


    ddf = dask_cudf.concat(results).compute()
    """
