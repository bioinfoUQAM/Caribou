#!/usr/bin python3

import argparse

from utils import *
from time import time
from pathlib import Path
from models.reads_simulation import split_sim_dataset
from models.classification import ClassificationMethods

__author__ = "Nicolas de Montigny"

__all__ = ['bacteria_extraction_train_cv']

TRAINING_DATASET_NAME = 'train'
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

    if opt['model_type'] != 'onesvm':
        if opt['data_host'] is not None:
            if opt['merged'] is not None:
                db_data, db_ds = verify_load_db(opt['merged'])
            else:
                db_data, db_ds = verify_load_host_merge(opt['data_bacteria'], opt['data_host'])
            db_name = 'host_merged'
        else:
            db_data, db_ds = verify_load_db(opt['data_bacteria'])
            db_name = opt['database_name']

        if opt['test'] is not None:
            test_data, test_ds = verify_load_db(opt['test'])
        else:
            test_data, test_ds = split_sim_dataset(db_ds, db_data, f'{TEST_DATASET_NAME}_{db_name}')
        if opt['validation'] is not None:
            val_data, val_ds = verify_load_db(opt['validation'])
        else:
            val_data, val_ds = split_sim_dataset(db_ds, db_data, f'{VALIDATION_DATASET_NAME}_{db_name}')
    else:
        if opt['merged'] is not None:
            db_data, db_ds = verify_load_db(opt['merged'])
        else:
            db_data, db_ds = verify_load_host_merge(opt['data_bacteria'], opt['data_host'])
        db_name = 'host_merged'

        if opt['test'] is not None:
            test_data, test_ds = verify_load_db(opt['test'])
        else:
            test_data, test_ds = split_sim_dataset(db_ds, db_data, f'{TEST_DATASET_NAME}_{db_name}')
        if opt['validation'] is not None:
            val_data, val_ds = verify_load_db(opt['validation'])
        else:
            val_data, val_ds = split_sim_dataset(db_ds, db_data, f'{VALIDATION_DATASET_NAME}_{db_name}')

        db_data, db_ds = verify_load_db(opt['data_bacteria'])
        db_name = opt['database_name']

    # Verify need for scaling
    scaling = verify_need_scaling(db_data)

    datasets = {
        TRAINING_DATASET_NAME : db_ds,
        TEST_DATASET_NAME : test_ds,
        VALIDATION_DATASET_NAME : val_ds
    }

# Training and cross-validation of models for bacteria extraction / host removal
################################################################################
    

    clf = ClassificationMethods(
        db_data = db_data,
        outdirs = outdirs,
        db_name = opt['database_name'],
        clf_binary = opt['model_type'],
        taxa = 'domain',
        batch_size = opt['batch_size'],
        training_epochs = opt['training_epochs'],
        scaling = scaling
    )

    t_s = time()

    cv_scores = clf.cross_validation(datasets)

    t_clf = time() - t_s
    print(f"Caribou finished training and cross-validating the {opt['model_type']} model in {t_clf} seconds")

# Argument parsing from CLI
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script trains and cross-validates a model for the bacteria extraction / host removal step.')
    # Database
    parser.add_argument('-db','--data_bacteria', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the bacteria database')
    parser.add_argument('-dh','--data_host', default=None, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the host')
    parser.add_argument('-dn','--database_name', required=True, help='Name of the bacteria database used to name files')
    parser.add_argument('-hn','--host_name', default=None, help='Name of the host database used to name files')
    # Optional datasets
    parser.add_argument('-m','--merged', default=None, type=Path, help='PATH to a npz file containing the k-mers profile for the merged bacteria and host databases')
    parser.add_argument('-v','--validation', default=None, type=Path, help='PATH to a npz file containing the k-mers profile for the validation dataset')
    parser.add_argument('-t','--test', default=None, type=Path, help='PATH to a npz file containing the k-mers profile for the test dataset')
    # Parameters
    parser.add_argument('-model','--model_type', required=True, choices=['onesvm','linearsvm','attention','lstm','deeplstm'], help='The type of model to train')
    parser.add_argument('-bs','--batch_size', default=32, type=int, help='Size of the batch size to use, defaults to 32')
    parser.add_argument('-e','--training_epochs', default=100, type=int, help='The number of training iterations for the neural networks models if one is chosen, defaults to 100')
    parser.add_argument('-o','--outdir', required=True, type=Path, help='PATH to a directory on file where outputs will be saved')
    parser.add_argument('-wd','--workdir', default='/tmp/spill', type=Path, help='Optional. Path to a working directory where Ray Tune will output and spill tuning data')
    args = parser.parse_args()

    opt = vars(args)

    bacteria_extraction_train_cv(opt)
