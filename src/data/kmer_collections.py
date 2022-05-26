import os
import vaex
import glob
import warnings

from shutil import rmtree
from subprocess import run
from joblib import Parallel, delayed, parallel_backend

import numpy as np
import pandas as pd

__author__ = ['Amine Remita', 'Nicolas de Montigny']

__all__ = ['kmers_collection','construct_data',
            'compute_seen_kmers_of_sequence','compute_given_kmers_of_sequence','compute_kmers',
            'parallel_extraction','build_kmers_Xy_data','build_kmers_X_data']

"""
Module adapted from module kmer_collections.py of
mlr_kgenomvir package [Remita et al. 2022]

Save kmers directly to drive instead of memory and adapted / added functions to do so.
Converted to be only functions instead of object for parallelization.
"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# #####
# Data build functions
# ####################

def build_kmers_Xy_data(seq_data, k, Xy_file, dataset, length = 0, kmers_list = None):

    if isinstance(kmers_list, list):
        method = 'given'
    else:
        method = 'seen'

    kmers_collection(seq_data, Xy_file, length, k, dataset, method = method, kmers_list = kmers_list)

    classes = np.array(seq_data.labels)

    return classes

def build_kmers_X_data(seq_data, X_file, kmers_list, k, dataset, length = 0):

    kmers_collection(seq_data, X_file, length, k, dataset, method = 'given', kmers_list = kmers_list)

# #####
# Kmers computing
# ##################

def kmers_collection(seq_data, Xy_file, length, k, dataset, method = 'seen', kmers_list = None):

    dir_path = os.path.join(os.path.split(Xy_file)[0],"tmp","")
    kmc_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"KMC","bin")
    faSplit = os.path.join(os.path.dirname(os.path.realpath(__file__)),"faSplit")

    compute_kmers(seq_data, method, kmers_list, k, dir_path, faSplit, kmc_path, Xy_file, dataset)

    rmtree(dir_path)

def construct_data(Xy_file, dir_path):
    df = None

    # Get all files in a list
    files_list = glob.glob(os.path.join(dir_path,'*.csv'))
    nb_files = len(files_list)

    if nb_files <= 1024:
        # If less than 1024 files -> open them all at ounce
        df = vaex.open_many(files_list)
    else:
        # Read + concat files in batch of 1024
        length_last_round = nb_files % 1024
        last_pos = 0
        while last_pos + length_last_round < nb_files:
            subset = files_list[last_pos:last_pos + 1024]
            if df is None:
                df = vaex.open_many(subset)
            else:
                df = df.concat(vaex.open_many(subset), resolver = 'flexible')
                last_pos = last_pos + 1024
        subset = files_list[last_pos:last_pos + length_last_round]
        df = df.concat(vaex.open_many(subset), resolver = 'flexible')

    colnames = list(df.columns)
    colnames.remove('id')
    # Fill NAs with 0
    df = df.fillna(0, column_names = colnames)
    # Remove columns filled with 0
    for col in colnames:
        if int(df.sum(col)) == 0:
            df = df.drop(col)
    df = df.extract()
    # Save dataframe
    df.export_hdf5(Xy_file)

def compute_seen_kmers_of_sequence(kmc_path, k, dir_path, ind, file):
    # Make tmp folder per sequence
    tmp_folder = os.path.join(dir_path,"tmp_{}".format(ind))
    id = os.path.splitext(os.path.basename(file))[0]
    os.mkdir(tmp_folder)
    # Count k-mers with KMC
    cmd_count = os.path.join(kmc_path,"kmc -k{} -fm -ci4 -cs1000000000 -m10 -hp {} {} {}".format(k, file, os.path.join(tmp_folder, str(ind)), tmp_folder))
    run(cmd_count, shell = True, capture_output=True)
    # Transform k-mers db with KMC
    cmd_transform = os.path.join(kmc_path,"kmc_tools transform {} dump {}".format(os.path.join(tmp_folder, str(ind)), os.path.join(dir_path, "{}.txt".format(ind))))
    run(cmd_transform, shell = True, capture_output=True)

    # Transpose kmers profile with pandas
    tmp_df = pd.read_table(os.path.join(dir_path,"{}.txt".format(ind)), sep = '\t', header = None, names = ['id', str(id)])
    tmp_df.T.to_csv(os.path.join(dir_path,"{}.csv".format(ind)), header = False)

    # Delete temp dir and file
    rmtree(tmp_folder)
    os.remove(os.path.join(dir_path,"{}.txt".format(ind)))

def compute_given_kmers_of_sequence(kmers_list, kmc_path, k, dir_path, ind, file):
    # Make tmp folder per sequence
    tmp_folder = os.path.join(dir_path,"tmp_{}".format(ind))
    id = os.path.splitext(os.path.basename(file))[0]
    os.mkdir(tmp_folder)
    # Count k-mers with KMC
    cmd_count = os.path.join(kmc_path,"kmc -k{} -fm -ci4 -cs1000000000 -m10 -hp {} {} {}".format(k, file, os.path.join(tmp_folder, str(ind)), tmp_folder))
    run(cmd_count, shell = True, capture_output=True)
    # Transform k-mers db with KMC
    cmd_transform = os.path.join(kmc_path,"kmc_tools transform {} dump {}".format(os.path.join(tmp_folder, str(ind)), os.path.join(dir_path, "{}.txt".format(ind))))
    run(cmd_transform, shell = True, capture_output=True)

    profile = pd.read_table(os.path.join(dir_path,"{}.txt".format(ind)), sep = '\t', header = None, names = ['id', str(id)]).T
    # Temp pandas df to write given kmers to file
    tmp_df = pd.DataFrame(np.zeros((1,len(kmers_list))), columns = kmers_list, index = [id])
    for kmer in kmers_list:
        if kmer in profile.columns:
            tmp_df.at[id,kmer] = profile.loc[id,kmer]
        else:
            tmp_df.at[id,kmer] = 0

    tmp_df.to_csv(os.path.join(dir_path,"{}.csv".format(ind)), header = False, index_label = 'id')

    # Delete temp dir and file
    rmtree(tmp_folder)
    os.remove(os.path.join(dir_path,"{}.txt".format(ind)))

def compute_kmers(seq_data, method, kmers_list, k, dir_path, faSplit, kmc_path, Xy_file, dataset):
    file_list = []

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    cmd_split = '{} byname {} {}'.format(faSplit, seq_data.data, dir_path)

    os.system(cmd_split)

    for id in seq_data.ids:
        file = os.path.join(dir_path,'{}.fa'.format(id))
        file_list.append(file)

    #Extract kmers in parallel using KMC3
    parallel_extraction(file_list, method, kmers_list, kmc_path, k, dir_path)
    # build kmers matrix
    construct_data(Xy_file, dir_path)


def parallel_extraction(file_list, method, kmers_list, kmc_path, k, dir_path):
    if method == 'seen':
        with parallel_backend('threading'):
            Parallel(n_jobs = -1, prefer = 'threads', verbose = 100)(
            delayed(compute_seen_kmers_of_sequence)
            (kmc_path, k, dir_path, i, file) for i, file in enumerate(file_list))
    elif method == 'given':
        with parallel_backend('threading'):
            Parallel(n_jobs = -1, prefer = 'threads', verbose = 100)(
            delayed(compute_given_kmers_of_sequence)
            (kmers_list, kmc_path, k, dir_path, i, file) for i, file in enumerate(file_list))
