#!/usr/bin python3

import argparse

from utils import *
from time import time
from pathlib import Path
from models.reads_simulation import split_sim_dataset
from models.classification_old import ClassificationMethods

__author__ = "Nicolas de Montigny"

__all__ = ['bacteria_extraction_train_cv']

VALIDATION_DATASET_NAME = 'validation'
TEST_DATASET_NAME = 'test'

# Initialisation / validation of parameters from CLI
################################################################################
def bacteria_extraction_train_cv(opt):

    # Validate training parameters
    verify_positive_int(opt['batch_size'], 'batch_size')
    verify_positive_int(opt['training_epochs'], 'number of iterations in neural networks training')
    
    outdirs = define_create_outdirs(opt['outdir'])
    
    # Initialize cluster
    init_ray_cluster(opt['workdir'])

# Data loading
################################################################################

    if opt['data_host'] is not None:
        db_data, db_ds = verify_load_host_merge(opt['data_bacteria'], opt['data_host'])
        db_name = 'host_merged'
    else:
        db_data, db_ds = verify_load_db(opt['data_bacteria'])
        db_name = opt['dataset_name']

    k_length = len(db_data['kmers'][0])

    test_ds, test_data = split_sim_dataset(db_ds, db_data, f'{TEST_DATASET_NAME}_{db_name}')
    val_ds, val_data = split_sim_dataset(db_ds, db_data, f'{VALIDATION_DATASET_NAME}_{db_name}')

# Training and cross-validation of models for bacteria extraction / host removal
################################################################################
    
    t_start = time()

    if opt['host_name'] is None:
        ClassificationMethods(
            database_k_mers = (db_data, db_ds),
            k = k_length,
            outdirs = outdirs,
            database = opt['database_name'],
            classifier_binary = opt['model_type'],
            taxa = 'domain',
            batch_size = opt['batch_size'],
            training_epochs = opt['training_epochs'],
            verbose = opt['verbose'],
            cv = True
        ).execute_training()
    else:
        ClassificationMethods(
            database_k_mers = (db_data, db_ds),
            k = k_length,
            outdirs = outdirs,
            database = opt['database_name'],
            classifier_binary = opt['model_type'],
            taxa = 'domain',
            batch_size = opt['batch_size'],
            training_epochs = opt['training_epochs'],
            verbose = opt['verbose'],
            cv = True
        ).execute_training()

    t_end = time()
    t_classify = t_end - t_start
    print(
        f"Caribou finished training and cross-validating the {opt['model_type']} model in {t_classify} seconds")


# Argument parsing from CLI
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script trains and cross-validates a model for the bacteria extraction / host removal step.')
    parser.add_argument('-db','--data_bacteria', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the bacteria database')
    parser.add_argument('-dh','--data_host', default=None, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the host')
    parser.add_argument('-dt','--database_name', required=True, help='Name of the bacteria database used to name files')
    parser.add_argument('-ds','--host_name', default=None, help='Name of the host database used to name files')
    parser.add_argument('-model','--model_type', required = True, choices=['onesvm','linearsvm','attention','lstm','deeplstm'], help='The type of model to train')
    parser.add_argument('-bs','--batch_size', default=32, type=int, help='Size of the batch size to use, defaults to 32')
    parser.add_argument('-e','--training_epochs', default=100, type=int, help='The number of training iterations for the neural networks models if one is chosen, defaults to 100')
    parser.add_argument('-v','--verbose', action='store_true', help='Should the program be verbose')
    parser.add_argument('-o','--outdir', required=True, type=Path, help='PATH to a directory on file where outputs will be saved')
    parser.add_argument('-wd','--workdir', default='/tmp/spill', type=Path, help='Optional. Path to a working directory where Ray Tune will output and spill tuning data')
    args = parser.parse_args()

    opt = vars(args)

    bacteria_extraction_train_cv(opt)
