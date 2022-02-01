from Caribou.data.seq_collections import SeqCollection

import re
import os
import time
import gzip

from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from Bio import SeqIO
from os.path import splitext
from subprocess import run
from shutil import rmtree

from joblib import Parallel, parallel_backend, wrap_non_picklable_objects
from dask.distributed import Client, LocalCluster

from scipy.sparse import csr_matrix, csc_matrix
from sklearn.feature_selection import VarianceThreshold
from tensorflow.config import list_physical_devices

import numpy as np
import pandas as pd
import tables as tb

import dask
import joblib

# From mlr_kgenomvir
__author__ = ['Amine Remita', 'Nicolas de Montigny']

__all__ = ['kmers_collection','construct_data','compute_seen_kmers_of_sequence','compute_given_kmers_of_sequence',
           'compute_kmers','loky','multiprocessing','threading','joblib_dask','dask_client','numba_parallel',
           'build_kmers_Xy_data','build_kmers_X_data']

"""
Module adapted from module kmer_collections.py of
mlr_kgenomvir package [Remita et al. 2021]

Save kmers directly to drive instead of memory and
adapted / added functions to do so.
Converted to be only functions instead of object for parallelization.
"""

# #####
# Data build functions
# ####################

def build_kmers_Xy_data(seq_data, k, Xy_file, length = 0, kmers_list = None):

    if kmers_list is not None:
        collection = kmers_collection(seq_data, Xy_file, length, k, method = 'given', kmers_list = kmers_list)
    else:
        collection = kmers_collection(seq_data, Xy_file, length, k, method = 'seen', kmers_list = None)

    kmers_list = collection['kmers_list']
    X_data = collection['data']
    y_data = np.array(seq_data.labels)
    ids = seq_data.ids

    return X_data, y_data, kmers_list

def build_kmers_X_data(seq_data, X_file, kmers_list, k, length = 0):

    collection = kmers_collection(seq_data, Xy_file, length, k, method = 'given', kmers_list = kmers_list)
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
    seq_data = seq_data
    Xy_file = Xy_file
    length = length
    k = k
    method = method
    kmers_list = kmers_list
    dir_path = os.path.split(Xy_file)[0] + "/tmp/"
    Xy_file = tb.open_file(Xy_file, "w")
    dict_data = defaultdict(lambda: [0]*length)
    kmc_path = "{}/KMC/bin".format(os.path.dirname(os.path.realpath(__file__)))
    faSplit = "{}/faSplit".format(os.path.dirname(os.path.realpath(__file__)))
    #
    dict_data = compute_kmers(seq_data, method, dict_data, kmers_list, k, dir_path, faSplit, kmc_path)
    if method == 'seen':
        collection['kmers_list'] = list(dict_data)
    elif method == 'given':
        collection['kmers_list'] = kmers_list
    collection['data'] = construct_data(dict_data, Xy_file)
    Xy_file.close()

    return collection

def construct_data(dict_data, Xy_file):
    # Convert dict_data to numpy array and write directly to didk with pytables
    data = Xy_file.create_carray("/", "data", obj = np.array([ dict_data[x] for x in dict_data ],dtype=np.uint64).T)
    return data

def compute_seen_kmers_of_sequence(dict_data, kmc_path, k, dir_path, ind, file):
    # Count k-mers with KMC
    cmd_count = "{}/kmc -k{} -fm -cs1000000000 -t68 -hp {} {}/{} {}".format(kmc_path, k, file, dir_path, ind, dir_path)
    #/localscratch/nicdemon.2258390.0/env/lib/python3.8/site-packages/Caribou/data/KMC/bin/kmc -k35 -fm -cs1000000000 -t68 -hp -sm $SLURM_TMPDIR/output/mock/data/tmp/NC_014830.1.fa $SLURM_TMPDIR/output/mock/data/tmp/0 $SLURM_TMPDIR/output/mock/data/tmp/
    run(cmd_count, shell = True, capture_output=True)
    # Transform k-mers db with KMC
    cmd_transform = "{}/kmc_tools transform {}/{} dump {}/{}.txt".format(kmc_path, dir_path, ind, dir_path, ind)
    #/localscratch/nicdemon.2258390.0/env/lib/python3.8/site-packages/Caribou/data/KMC/bin/kmc_tools transform $SLURM_TMPDIR/output/mock/data/tmp/0 dump $SLURM_TMPDIR/output/mock/data/tmp/0.txt
    run(cmd_transform, shell = True, capture_output=True)
    # Parse k-mers file to pandas
    profile = np.loadtxt('{}/{}.txt'.format(dir_path, ind), dtype = object)
    # Save to Xyfile
    for row in profile:
        dict_data[row[0]][ind] = int(row[1])
    return dict_data

def compute_given_kmers_of_sequence(dict_data, kmers_list, kmc_path, k, dir_path, ind, file):
    # Count k-mers with KMC
    cmd_count = "{}/kmc -k{} -fm -cs1000000000 -t68 -hp {} {}/{} {}".format(kmc_path, k, file, dir_path, ind, dir_path)
    run(cmd_count, shell = True, capture_output=True)
    # Transform k-mers db with KMC
    cmd_transform = "{}/kmc_tools transform {}/{} dump {}/{}.txt".format(kmc_path, dir_path, ind, dir_path, ind)
    run(cmd_transform, shell = True, capture_output=True)
    # Parse k-mers file to pandas
    profile = np.loadtxt('{}/{}.txt'.format(dir_path, ind), dtype = object)

    for kmer in kmers_list:
        ind_kmer = kmers_list.index(kmer)
        for row in profile:
            if row[0] == kmer:
                dict_data[row[0]][ind] = int(row[1])
            else:
                dict_data[row[0]][ind] = 0

    return dict_data

def compute_kmers(seq_data, method, dict_data, kmers_list, k, dir_path, faSplit, kmc_path):
    path, ext = splitext(seq_data.data)
    ext = ext.lstrip(".")
    file_list = []

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    cmd_split = '{} byname {} {}'.format(faSplit, seq_data.data, dir_path)

    os.system(cmd_split)

    for id in seq_data.ids:
        file = dir_path + id + '.fa'
        file_list.append(file)

    try:
        t_start = time.time()
        loky(file_list, method, dict_data, kmers_list, kmc_path, k, dir_path)
        print("Joblib loky : {:.3f}s".format(time.time() - t_start))
        rmtree(dir_path)
        os.mkdir(dir_path)
        dict_data = {}
    except Error as e:
        print("Joblib loky did not work")
        print(e)
    try:
        t_start = time.time()
        threading(file_list, method, dict_data, kmers_list, kmc_path, k, dir_path)
        print("Joblib threading : {:.3f}s".format(time.time() - t_start))
        rmtree(dir_path)
        os.mkdir(dir_path)
        dict_data = {}
    except Error as e:
        print("Joblib threading did not work")
        print(e)
    try:
        t_start = time.time()
        joblib_dask(file_list, method, dict_data, kmers_list, kmc_path, k, dir_path)
        print("Joblib dask : {:.3f}s".format(time.time() - t_start))
        rmtree(dir_path)
        os.mkdir(dir_path)
        dict_data = {}
    except Error as e:
        print("Joblib dask did not work")
        print(e)
    try:
        t_start = time.time()
        dask_client(file_list, method, dict_data, kmers_list, kmc_path, k, dir_path)
        print("Dask client : {:.3f}s".format(time.time() - t_start))
        rmtree(dir_path)
        os.mkdir(dir_path)
        dict_data = {}
    except Error as e:
        print("Dask client did not work")
        print(e)

    """
    # Detect if a GPU is available
    if list_physical_devices('GPU'):
        dict_data = dask(file_list, fx)
    else:
        dict_data = numba(file_list)
        #loky(file_list, fx)

    rmtree(dir_path)
    """
    return dict_data

def loky(file_list, method, dict_data, kmers_list, kmc_path, k, dir_path):
    if method == 'seen':
        with parallel_backend('loky'):
            dict_data = Parallel(n_jobs = -1, prefer = 'processes', verbose = 100)(
            joblib.delayed(compute_seen_kmers_of_sequence)
            (dict_data, kmc_path, k, dir_path, i, file) for i, file in enumerate(file_list))
    elif method == 'given':
        with parallel_backend('loky'):
            dict_data = Parallel(n_jobs = -1, prefer = 'processes', verbose = 100)(
            joblib.delayed(compute_given_kmers_of_sequence)
            (dict_data, kmers_list, kmc_path, k, dir_path, i, file) for i, file in enumerate(file_list))

    return dict_data

def threading(file_list, method, dict_data, kmers_list, kmc_path, k, dir_path):
    if method == 'seen':
        with parallel_backend('threading'):
            dict_data = Parallel(n_jobs = -1, prefer = 'processes', verbose = 100)(
            joblib.delayed(compute_seen_kmers_of_sequence)
            (dict_data, kmc_path, k, dir_path, i, file) for i, file in enumerate(file_list))
    elif method == 'given':
        with parallel_backend('threading'):
            dict_data = Parallel(n_jobs = -1, prefer = 'processes', verbose = 100)(
            joblib.delayed(compute_given_kmers_of_sequence)
            (dict_data, kmers_list, kmc_path, k, dir_path, i, file) for i, file in enumerate(file_list))

    return dict_data

def joblib_dask(file_list, method, dict_data, kmers_list, kmc_path, k, dir_path):
    client = Client(processes=False)
    if method == 'seen':
        with parallel_backend('dask'):
            dict_data = Parallel(n_jobs = -1, prefer = 'processes', verbose = 100)(
            joblib.delayed(compute_seen_kmers_of_sequence)
            (dict_data, kmc_path, k, dir_path, i, file) for i, file in enumerate(file_list))
    elif method == 'given':
        with parallel_backend('dask'):
            dict_data = Parallel(n_jobs = -1, prefer = 'processes', verbose = 100)(
            joblib.delayed(compute_given_kmers_of_sequence)
            (dict_data, kmers_list, kmc_path, k, dir_path, i, file) for i, file in enumerate(file_list))

    return dict_data

def dask_client(file_list, method, dict_data, kmers_list, kmc_path, k, dir_path):
    #See on Narval if detects well and need to specify nb of threads
    #cluster = LocalCluster(processes = True, n_workers = 6, threads_per_worker = 68)
    cluster = LocalCluster(processes = True)
    client = Client(cluster)
    print("Client : ", client)
    jobs = []

    if method == 'seen':
        for i, file in enumerate(file_list):
            job = client.submit(compute_seen_kmers_of_sequence, dict_data, kmc_path, k, dir_path, i, file, pure = False)
            jobs.append(job)
    elif method == 'given':
        for i, file in enumerate(file_list):
            job = client.submit(compute_given_kmers_of_sequence, dict_data, kmers_list, kmc_path, k, dir_path, i, file, pure = False)
            jobs.append(job)

    results = client.gather(jobs)

    for result in results:
        dict_data = {**result, **dict_data}

    client.close()
    return dict_data
