#!/usr/bin python3

import ray
import os.path
import argparse

from utils import *
from time import time
from pathlib import Path

from data.reduction.chi2_selection import TensorChi2Selection
from data.reduction.occurence_exclusion import TensorPercentOccurenceExclusion

__author__ = "Nicolas de Montigny"

__all__ = ['features_reduction']

"""
This script computes features reduction to a given K-mers dataset and then applies it.
The method is based on the KRFE algorithm (Lebatteux et al., 2019)
"""

# Initialisation / validation of parameters from CLI
################################################################################
def features_reduction(opt):

    # Verify existence of files and load data
    data = verify_load_data(opt['dataset'])
    k_length = len(data['kmers'][0])
    verify_file(opt['kmers_list'])

    # Verification of k length
    k_length, kmers_list = verify_kmers_list_length(k_length, opt['kmers_list'])

    # Not sure if needed for training KRFE
    """
    # Verify that model type is valid / choose default depending on host presence
    if opt['model_type'] is None:
        opt['model_type'] = 'cnn'
    """
    outdirs = define_create_outdirs(opt['outdir'])
    
    # Initialize cluster
    init_ray_cluster(opt['workdir'])

# Features reduction
################################################################################
    """
    Brute force -> Features statistically related to classes
    1. OccurenceExclusion (5% extremes)
    2. Chi2 + SelectKBest() (<0.05 p-value)
    """

    # Load data 
    ds = ray.data.read_parquet(data['profile'])
    # Iterate over methods for exp results
    t_start = time()
    ds, kmers_list = occurence_exclusion(ds, kmers_list)
    ds, data['kmers'] = chi2selection(ds, kmers_list)
    t_end = time()
    t_reduction = t_end - t_start
    # Save reduced dataset
    data['profile'] = f"{data['profile']}_reduced"
    ds.write_parquet(data['profile'])
    # Save reduced K-mers
    with open(os.path.join(outdirs["data_dir"],'kmers_list.txt'),'w') as handle:
        handle.writelines("%s\n" % item for item in data['kmers'])
    # Save reduced data
    path, ext = os.path.splitext(opt['dataset'])
    data_file = f'{path}_reduced{ext}'
    save_Xy_data(data, data_file)

    print(f"Caribou finished reducing k-mers features of {opt['dataset_name']} using the combined occurence and chi2 methods from the original dataset in {t_reduction} seconds.")

# Exclusion columns occuring in less / more than 10% of the columns
def occurence_exclusion(ds, kmers):
    preprocessor = TensorPercentOccurenceExclusion(
        features = kmers,
        percent = 0.05
    )
    
    ds = preprocessor.fit_transform(ds)
    
    kmers = preprocessor.stats_['cols_keep']

    return ds, kmers

# Chi2 evaluation of dependance between features and classes
def chi2selection(ds, kmers):
    preprocessor = TensorChi2Selection(
        features = kmers,
        threshold = 0.05
    )
    
    ds = preprocessor.fit_transform(ds)

    kmers = preprocessor.stats_['cols_keep']

    return ds, kmers

# Argument parsing from CLI
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script computes features reduction to a given K-mers dataset and then applies it.')
    # Dataset
    parser.add_argument('-db','--dataset', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the bacteria database')
    parser.add_argument('-dt','--dataset_name', default='dataset', help='Name of the dataset used to name files')
    parser.add_argument('-l','--kmers_list', default=None, type=Path, help='PATH to a file containing a list of k-mers that will be reduced')
    # Parameters
    parser.add_argument('-o','--outdir', required=True, type=Path, help='PATH to a directory on file where outputs will be saved')
    parser.add_argument('-wd','--workdir', default='/tmp/spill', type=Path, help='Optional. Path to a working directory where tuning data will be spilled')
    args = parser.parse_args()

    opt = vars(args)

    features_reduction(opt)
