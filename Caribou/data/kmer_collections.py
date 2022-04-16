import os
import glob
import warnings

from subprocess import run
from shutil import rmtree
from itertools import product
from collections import defaultdict

from joblib import Parallel, delayed, parallel_backend
from tensorflow.config import list_physical_devices

import numpy as np
import tables as tb
import pandas as pd
import vaex

# Sparse matrix pour -> numpy (comment gérer memoire)
# Possibilité d'enlever des k-mers peu représentés dans sequences (k-mers pertinents != nécessairement discriminant/minimum) (bags of k-mers)
    # Stratégie pour déterminer quels à enlever
    # K-mers composite -> batch (+) représentants (kmers dégénérés)
# Étape initiale de sampling de k-mers (représentatif pas trop grand) -> Kevolve-like pr choisir les kmers importants (seed), pls en // -> puis extrait seulement les kmers selected par la suite
    # Voir avec Dylan algo ~ Kevolve

# From mlr_kgenomvir
__author__ = ['Amine Remita', 'Nicolas de Montigny']

__all__ = ['kmers_collection','construct_data','save_kmers_profile','save_id_file_list',
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

    if kmers_list is not None:
        method = 'given'
    else:
        method = 'seen'

    collection = kmers_collection(seq_data, Xy_file, length, k, dataset, method = method, kmers_list = kmers_list)

    kmers_list = collection['kmers_list']
    X_data = collection['data']
    y_data = np.array(seq_data.labels)
    ids = seq_data.ids

    return X_data, y_data, kmers_list

def build_kmers_X_data(seq_data, X_file, kmers_list, k, dataset, length = 0):

    collection = kmers_collection(seq_data, X_file, length, k, dataset, method = 'given', kmers_list = kmers_list)
    kmers_list = collection['kmers_list']
    X_data = collection['data']
    ids = seq_data.ids

    return X_data, kmers_list, ids

# #####
# Kmers computing
# ##################

def kmers_collection(seq_data, Xy_file, length, k, dataset, method = 'seen', kmers_list = None):
    collection = {}
    #
    collection['data'] = Xy_file
    dir_path = os.path.split(Xy_file)[0] + "/tmp/"
    kmc_path = "{}/KMC/bin".format(os.path.dirname(os.path.realpath(__file__)))
    faSplit = "{}/faSplit".format(os.path.dirname(os.path.realpath(__file__)))
    #
    collection['ids'], collection['kmers_list'] = compute_kmers(seq_data, method, kmers_list, k, dir_path, faSplit, kmc_path, Xy_file, dataset)
    #
    rmtree(dir_path)

    return collection

def construct_data(Xy_file, dir_path, list_id_file):
    ids = [id for id,file in list_id_file]
    df = None
    ids = []
    # Iterate over ids / files
    for id, file in list_id_file:
        if df is None:
            df = vaex.read_csv(file, sep = '\t', header = None, names = ['kmers', id])
        else:
            try:
                # Read each file individually
                tmp = vaex.read_csv(file, sep = '\t', header = None, names = ['kmers', id])
                # Join each files to the previously computed dataframe
                df = df.join(tmp, on='kmers', how = 'left')
                ids.append(id)
            except ValueError:
                print("Identical sequence IDs not supported, every sequence should have a unique ID")

    # Extract k-mers list
    kmers_list = list(df.kmers.values)
    # Fill NAs with 0
    df = df.fillna(0)
    # Convert to numpy array to transpose and reconvert to vaex df
    df = np.array(df.to_arrays(column_names = ids, array_type = 'numpy'))

    save_kmers_profile(df, Xy_file, tmp = False)

    return ids, kmers_list

def save_kmers_profile(df, Xy_file, tmp = True):
    # Convert vaez dataframe to numpy array and write directly to disk with pytables
    with tb.open_file(Xy_file, "w") as handle:
        data = handle.create_carray("/", "data", obj = df)

def compute_seen_kmers_of_sequence(kmc_path, k, dir_path, ind, file):
    if not os.path.isfile('{}/{}.csv'.format(dir_path, ind)):
        # Make tmp folder per sequence
        tmp_folder = "{}tmp_{}/".format(dir_path, ind)
        id = os.path.splitext(os.path.basename(file))[0]
        try:
            os.mkdir(tmp_folder)
            # Count k-mers with KMC
            cmd_count = "{}/kmc -k{} -fm -ci1 -cs1000000000 -m10 -hp {} {}/{} {}".format(kmc_path, k, file, tmp_folder, ind, tmp_folder)
            run(cmd_count, shell = True, capture_output=True)
            # Transform k-mers db with KMC
            cmd_transform = "{}/kmc_tools transform {}/{} dump {}/{}.txt".format(kmc_path, tmp_folder, ind, dir_path, ind)
            run(cmd_transform, shell = True, capture_output=True)
        except:
            pass

        return id, "{}/{}.txt".format(dir_path, ind)

def compute_given_kmers_of_sequence(kmers_list, kmc_path, k, dir_path, ind, file):
    # Make tmp folder per sequence
    tmp_folder = "{}tmp_{}".format(dir_path, ind)
    id = os.path.splitext(os.path.basename(file))[0]
    try:
        os.mkdir(tmp_folder)
        # Count k-mers with KMC
        cmd_count = "{}/kmc -k{} -fm -ci1 -cs1000000000 -m10 -hp {} {}/{} {}".format(kmc_path, k, file, tmp_folder, ind, tmp_folder)
        run(cmd_count, shell = True, capture_output=True)
        # Transform k-mers db with KMC
        cmd_transform = "{}/kmc_tools transform {}/{} dump {}/{}.txt".format(kmc_path, tmp_folder, ind, dir_path, ind)
        run(cmd_transform, shell = True, capture_output=True)
    except:
        pass

    try:
        profile = pd.read_table('{}/{}.txt'.format(dir_path, ind), names = [id], index_col = 0, dtype = object).T
        # Temp pandas df to write given kmers to file
        df = pd.DataFrame(np.zeros((1,len(kmers_list))), columns = kmers_list, index = [id])

        for kmer in kmers_list:
            if kmer in profile.columns:
                df.at[id,kmer] = profile.loc[id,kmer]
            else:
                df.at[id,kmer] = 0

                df.T.to_csv('{}/{}.txt'.format(dir_path, ind), sep = "\t", header = ['kmers',id])
    except:
        print("Kmers extraction error for sequence {}".format(id))

    return id, '{}/{}.txt'.format(dir_path, ind)

def compute_kmers(seq_data, method, kmers_list, k, dir_path, faSplit, kmc_path, Xy_file, dataset):
    file_list_ids_file = os.path.join(os.path.dirname(Xy_file),'list_id_file_{}.txt'.format(dataset))
    if not os.path.isfile(file_list_ids_file):
        file_list = []

        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        cmd_split = '{} byname {} {}'.format(faSplit, seq_data.data, dir_path)

        os.system(cmd_split)

        for id in seq_data.ids:
            file = dir_path + id + '.fa'
            file_list.append(file)

        list_id_file= parallel_extraction(file_list, method, kmers_list, kmc_path, k, dir_path)
        save_id_file_list(list_id_file,file_list_ids_file)
        ids, kmers_list = construct_data(Xy_file, dir_path, list_id_file)

    else:
        with open(file_list_ids_file, 'r') as handle:
            list_id_file = [tuple(line.strip('\n').split(',')) for line in handle]
        # Detect if a GPU is available
        if len(list_physical_devices('GPU')) > 0:
            ids, kmers_list = construct_data_GPU(Xy_file, list_id_file, kmers_list)
        else:
            ids, kmers_list = construct_data(Xy_file, dir_path, list_id_file)

    os.remove(file_list_ids_file)

    return ids, kmers_list

def save_id_file_list(list_id_file, file):
    with open(file, 'w') as handle:
        for id, file in list_id_file:
            handle.write("{},{}\n".format(id,file))

def parallel_extraction(file_list, method, kmers_list, kmc_path, k, dir_path):
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
