#!/usr/bin python3

import ray
import os.path
import argparse

import numpy as np

from utils import *
from time import time
from glob import glob
from pathlib import Path


from data.reduction.low_var_selection import TensorLowVarSelection
from models.preprocessors.tfidf_transformer import TensorTfIdfTransformer
from data.reduction.chi_features_selection import TensorChiFeaturesSelection
from data.reduction.occurence_exclusion import TensorPercentOccurenceExclusion

__author__ = "Nicolas de Montigny"

__all__ = ['features_reduction']

"""
This script computes features reduction to a given K-mers dataset and then applies it.
"""

# Initialisation / validation of parameters from CLI
################################################################################
def features_reduction(opt):

    # Verify existence of files and load data
    data = verify_load_data(opt['dataset'])
    k_length = len(data['kmers'][0])
    verify_file(opt['kmers_list'])

    # Verification of k length
    k_length, kmers = verify_kmers_list_length(k_length, opt['kmers_list'])

    outdirs = define_create_outdirs(opt['outdir'])
    
    # Initialize cluster
    init_ray_cluster(opt['workdir'])

# Features reduction
################################################################################
    """
    Two-step features reduction :
    0. Features scaling
        1. TF-IDF scaling (diminish impact of more present and augment impact of less present)
    1. Brute force features exclusion
        1. OccurenceExclusion (exclusion of features present in more than 95% of samples)
        2. LowVarSelection (exclusion of features with less than 5% variance)
    2. Statistical features selection
        1. Chi2 + SelectPercentile() (select 25% of features with highest Chi2 values)
    3. In training features selection
        1. RandomForestClassification (select features identified as useful for classification)
        2. TruncatedSVD decomposition (map the features to 10 000 decomposed features if there is still more)
    """

    # Load data 
    files_lst = glob(os.path.join(data['profile'],'*.parquet'))
    export_ds = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))
    train_ds = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))
    # Time the computation of transformations
    t_start = time()
    # Features scaling
    train_ds = tfidf_transform(train_ds, kmers)
    # Brute force features exclusion
    train_ds, export_ds, kmers = occurence_exclusion(train_ds, export_ds, kmers)
    train_ds, export_ds, kmers = low_var_selection(train_ds, export_ds, kmers)
    # Statistical features selection
    train_ds, export_ds, data['kmers'] = features_selection(train_ds, export_ds, kmers, opt['taxa'])
    # Time the computation of transformations
    t_end = time()
    t_reduction = t_end - t_start
    # Save reduced dataset
    data['profile'] = f"{data['profile']}_reduced"
    export_ds.write_parquet(data['profile'])
    # Save reduced K-mers
    with open(os.path.join(outdirs["data_dir"],'kmers_list_reduced.txt'),'w') as handle:
        handle.writelines("%s\n" % item for item in data['kmers'])
    # Save reduced data
    path, ext = os.path.splitext(opt['dataset'])
    data_file = f'{path}_reduced{ext}'
    save_Xy_data(data, data_file)

    print(f"Caribou finished reducing k-mers features of {opt['dataset_name']} in {t_reduction} seconds.")

# TF-IDF scaling of the features
def tfidf_transform(ds, kmers):
    preprocessor = TensorTfIdfTransformer(
        features = kmers
    )
    ds = preprocessor.fit_transform(ds)

    return ds

# Exclusion of columns occuring in more than 95% of the samples
def occurence_exclusion(train_ds, export_ds, kmers):
    preprocessor = TensorPercentOccurenceExclusion(
        features = kmers,
        percent = 0.5
    )
    
    train_ds = preprocessor.fit_transform(train_ds)
    export_ds = preprocessor.transform(export_ds)
    kmers = preprocessor.stats_['cols_keep']

    return train_ds, export_ds, kmers

# Exclusion of columns with less than 5% variance
def low_var_selection(train_ds, export_ds, kmers):
    preprocessor = TensorLowVarSelection(
        features = kmers,
        threshold = 0.05,
    )

    train_ds = preprocessor.fit_transform(train_ds)
    export_ds = preprocessor.transform(export_ds)
    kmers = preprocessor.stats_['cols_keep']

    return train_ds, export_ds, kmers

# Chi2 evaluation of dependance between features and classes
# Select 25% of features with highest Chi2 values
def features_selection(train_ds, export_ds, kmers, taxa):
    preprocessor = TensorChiFeaturesSelection(
            features = kmers,
            taxa = taxa,
            threshold = 0.75,
        )

    train_ds = preprocessor.fit_transform(train_ds)
    export_ds = preprocessor.transform(export_ds)
    kmers = preprocessor.stats_['cols_keep']
    
    return train_ds, export_ds, kmers

# Argument parsing from CLI
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script computes features reduction to a given K-mers dataset and then applies it.')
    # Dataset
    parser.add_argument('-db','--dataset', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the bacteria database')
    parser.add_argument('-dt','--dataset_name', default='dataset', help='Name of the dataset used to name files')
    parser.add_argument('-l','--kmers_list', default=None, type=Path, help='PATH to a file containing a list of k-mers that will be reduced')
    # Parameters
    parser.add_argument('-t','--taxa', default='phylum', help='The taxonomic level to use for the classification, defaults to Phylum.')
    parser.add_argument('-o','--outdir', required=True, type=Path, help='PATH to a directory on file where outputs will be saved')
    parser.add_argument('-wd','--workdir', default='/tmp/spill', type=Path, help='Optional. Path to a working directory where tuning data will be spilled')
    args = parser.parse_args()

    opt = vars(args)

    features_reduction(opt)
