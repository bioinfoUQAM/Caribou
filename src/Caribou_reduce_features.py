#!/usr/bin python3

import ray
import os.path
import argparse

import numpy as np

from utils import *
from time import time
from glob import glob
from pathlib import Path

from models.preprocessors.tfidf_transformer import TensorTfIdfTransformer

from data.reduction.low_var_selection import TensorLowVarSelection
from data.reduction.chi_features_selection import TensorChiFeaturesSelection
from data.reduction.rdf_features_selection import TensorRDFFeaturesSelection
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

    outdirs = define_create_outdirs(opt['outdir'])
    
    # Initialize cluster
    init_ray_cluster(opt['workdir'])

# Features reduction
################################################################################
    """
    First option : Select features relevant to classification by Random Forest of decision trees
    
    Brute force -> Features statistically related to classes
    1. OccurenceExclusion (10% extremes)
    2. LowVarSelection (variance > 10%)
    3. Chi2 + SelectPercentile() (75% best values)
    """

    """
    TODO: Add to preprocessing in model training
    1. Replace the MinMaxScaling -> TfidfTransformer to scale down the impact of tokens that occur very frequently (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer)
    2. TruncatedSVD to reduce dimensions and keep 10 000 features ~PCA (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD)
    """

    # Load data 
    files_lst = glob(os.path.join(data['profile'],'*.parquet'))
    ds = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))
    # ds = ray.data.read_parquet(data['profile'], parallelism = -1)
    # Time the computation of transformations
    t_start = time()
    ds = tfidf_transform(ds, kmers_list)
    ds, kmers_list = tree_relevant_features(ds, kmers_list, 'phylum')
    print(len(kmers_list))
    if len(kmers_list) == 0:
        ds, kmers_list = occurence_exclusion(ds, opt['kmers_list'])
        ds, kmers_list = low_var_selection(ds,kmers_list)
        ds, data['kmers'] = features_selection(ds, kmers_list, 'phylum')
    t_end = time()
    t_reduction = t_end - t_start
    # Save reduced dataset
    data['profile'] = f"{data['profile']}_reduced"
    ds.write_parquet(data['profile'])
    # Save reduced K-mers
    with open(os.path.join(outdirs["data_dir"],'kmers_list_reduced.txt'),'w') as handle:
        handle.writelines("%s\n" % item for item in data['kmers'])
    # Save reduced data
    path, ext = os.path.splitext(opt['dataset'])
    data_file = f'{path}_reduced{ext}'
    save_Xy_data(data, data_file)

    print(f"Caribou finished reducing k-mers features of {opt['dataset_name']} in {t_reduction} seconds.")

# Exclusion of columns occuring in less / more than 10% of the columns = 20% removed
def occurence_exclusion(ds, kmers):
    preprocessor = TensorPercentOccurenceExclusion(
        features = kmers,
        percent = 0.1 # remove features present in less than 10% samples
    )
    
    ds = preprocessor.fit_transform(ds)
    kmers = preprocessor.stats_['cols_keep']

    return ds, kmers

# Exclusion of columns with less than 10% variance
def low_var_selection(ds, kmers):
    preprocessor = TensorLowVarSelection(
        features = kmers,
        threshold = 0.1, # remove features with less than 5% variance
    )

    ds = preprocessor.fit_transform(ds)
    kmers = preprocessor.stats_['cols_keep']

    return ds, kmers

# Chi2 evaluation of dependance between features and classes
def features_selection(ds, kmers, taxa):
    preprocessor = TensorChiFeaturesSelection(
            features = kmers,
            taxa = taxa,
            threshold = 0.75, # Keep 25% higest results
        )

    ds = preprocessor.fit_transform(ds)
    kmers = preprocessor.stats_['cols_keep']
    print(len(kmers))

    return ds, kmers

# TF-IDF scaling of the features
def tfidf_transform(ds, kmers):
    preprocessor = TensorTfIdfTransformer(
        features = kmers
    )
    ds = preprocessor.fit_transform(ds)

    return ds

# Decision tree feature selection to keep only those identified as relevant to classification
def tree_relevant_features(ds, kmers, taxa):
    preprocessor = TensorRDFFeaturesSelection(
        features = kmers,
        taxa = taxa
    )
    preprocessor.fit_transform(ds)

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
