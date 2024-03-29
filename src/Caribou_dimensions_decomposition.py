#!/usr/bin python3

import ray
import os.path
import argparse

import numpy as np

from utils import *
from time import time
from glob import glob
from pathlib import Path

from ray.data.preprocessors import Chain
from data.reduction.nmf_decomposition import TensorNMFDecomposition
from models.preprocessors.tfidf_transformer import TensorTfIdfTransformer
from data.reduction.dictionnary_decomposition import TensorDictionnaryDecomposition
from data.reduction.truncated_svd_decomposition import TensorTruncatedSVDDecomposition

__author__ = "Nicolas de Montigny"

__all__ = ['dimensions_decomposition']

"""
This script computes dimensions decomposition via TruncatedSVD and saves a reduced version of the dataset.
"""

# Initialisation / validation of parameters from CLI
################################################################################
def dimensions_decomposition(opt):
    
    # Verify existence of files and load data
    data = verify_load_data(opt['dataset'])

    # Verification of k length
    k_length = len(data['kmers'][0])
    verify_file(opt['kmers_list'])
    k_length, kmers = verify_kmers_list_length(k_length, opt['kmers_list'])

    outdirs = define_create_outdirs(opt['outdir'])

    # Initialize cluster
    init_ray_cluster(opt['workdir'])

# Dimensions decomposition
################################################################################

    # Define new file
    path, ext = os.path.splitext(opt['dataset'])
    data_file = f'{path}_decomposed{ext}'

    if not os.path.exists(data_file):
        if opt['nb_components'] < len(kmers):
            # Load data
            ds = read_parquet_files(data['profile'])

            scaler_file = os.path.join(outdirs['models_dir'], 'TF-IDF_diag.npz')
            reductor_file = os.path.join(outdirs['models_dir'], 'TruncatedSVD_components.npz')

            # Compute the decomposition
            preprocessor = Chain(
                TensorTfIdfTransformer(
                    features = kmers,
                    file = scaler_file
                ),
                TensorTruncatedSVDDecomposition(
                    features = kmers,
                    nb_components = opt['nb_components'],
                    file = reductor_file
                )
            )
            t_s = time()
            ds = preprocessor.fit_transform(ds)

            # Save decomposed dataset
            data['profile'] = f"{data['profile']}_decomposed"
            data['kmers'] = [f'feature_{i}' for i in np.arange(preprocessor.preprocessors[1]._nb_components)]
            ds.write_parquet(data['profile'])
            t_decomposition = time() - t_s

            # Save decomposed data
            save_Xy_data(data, data_file)

            print(f"Caribou finished decomposing the features in {t_decomposition} seconds.")
        else:
            print('Caribou did not decompose the features because the number to extract is bigger than the actual number of features')
    else:
        print("Caribou did not decompose the features because the file already exists")

# Argument parsing from CLI
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script computes features decomposition to a given K-mers dataset and then applies it.')
    # Dataset
    parser.add_argument('-db','--dataset', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the bacteria database')
    parser.add_argument('-l','--kmers_list', default=None, type=Path, help='PATH to a file containing a list of k-mers that will be reduced')
    # Parameters
    parser.add_argument('-n','--nb_components', default=1000, type=int, help='Number of components to decompose data into')
    parser.add_argument('-o','--outdir', required=True, type=Path, help='PATH to a directory on file where outputs will be saved')
    parser.add_argument('-wd','--workdir', default='/tmp/spill', type=Path, help='Optional. Path to a working directory where tuning data will be spilled')
    args = parser.parse_args()

    opt = vars(args)

    dimensions_decomposition(opt)

# Test params
opt = {
    'dataset':'/home/nicdemon/results/data/Xy_genome_cucurbita_data_K10.npz',
    'kmers_list':'/home/nicdemon/results/data/kmers_list_reduced.txt',
    'nb_components':10000,
    'outdir':'/home/nicdemon/results/',
    'workdir':'/home/nicdemon/ray/',
}
