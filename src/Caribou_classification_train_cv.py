#!/usr/bin python3

import warnings
import argparse

from utils import *
from pathlib import Path
from logging import ERROR
from models.classification import ClassificationMethods

warnings.filterwarnings('ignore')

__author__ = "Nicolas de Montigny"

__all__ = ['bacteria_classification_train_cv']

# Initialisation / validation of parameters from CLI
################################################################################
def bacteria_classification_train_cv(opt):
    # Verify existence of files and load data
    data_bacteria = verify_load_data(opt['data_bacteria'])
    k_length = len(data_bacteria['kmers'][0])

    # Verify that model type is valid / choose default depending on host presence
    if opt['model_type'] is None:
        opt['model_type'] = 'cnn'

    # Validate training parameters
    verify_positive_int(opt['batch_size'], 'batch_size')
    verify_positive_int(opt['training_epochs'], 'number of iterations in neural networks training')
    
    outdirs = define_create_outdirs(opt['outdir'])
    
    # Validate and extract list of taxas
    if opt['taxa'] is not None:
        lst_taxas = verify_taxas(opt['taxa'], data_bacteria['taxas'])
    else:
        lst_taxas = data_bacteria['taxas'].copy()
    
    if 'domain' in lst_taxas:
        lst_taxas.remove('domain')
    
    # Initialize cluster
    init_ray_cluster(opt['workdir'])

# Training and cross-validation of models for classification of bacterias
################################################################################
    ClassificationMethods(
        database_k_mers = data_bacteria,
        k = k_length,
        outdirs = outdirs,
        database = opt['database_name'],
        classifier_binary = None,
        classifier_multiclass = opt['model_type'],
        taxa = lst_taxas,
        batch_size = opt['batch_size'],
        training_epochs = opt['training_epochs'],
        verbose = opt['verbose'],
        cv = True
    ).execute_training()

    print(
        f"Caribou finished training and cross-validating the {opt['model_type']} model")

# Argument parsing from CLI
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script trains and cross-validates a model for the bacteria classification step.')
    parser.add_argument('-db','--data_bacteria', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the bacteria database')
    parser.add_argument('-dt','--database_name', required=True, help='Name of the bacteria database used to name files')
    parser.add_argument('-model','--model_type', default='lstm_attention', choices=['sgd','mnb','lstm_attention','cnn','widecnn'], help='The type of model to train')
    parser.add_argument('-t','--taxa', default=None, help='The taxonomic level to use for the classification, defaults to None. Can be one level or a list of levels separated by commas.')
    parser.add_argument('-bs','--batch_size', default=32, type=int, help='Size of the batch size to use, defaults to 32')
    parser.add_argument('-e','--training_epochs', default=100, type=int, help='The number of training iterations for the neural networks models if one ise chosen, defaults to 100')
    parser.add_argument('-v','--verbose', action='store_true', help='Should the program be verbose')
    parser.add_argument('-o','--outdir', required=True, type=Path, help='PATH to a directory on file where outputs will be saved')
    parser.add_argument('-wd','--workdir', default='/tmp/spill', type=Path, help='Optional. Path to a working directory where Ray Tune will output and spill tuning data')
    args = parser.parse_args()

    opt = vars(args)

    bacteria_classification_train_cv(opt)
