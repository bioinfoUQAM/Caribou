#!/usr/bin python3

import warnings
import argparse

from utils import *
from time import time
from pathlib import Path
from logging import ERROR
from models.reads_simulation import split_sim_dataset
from models.classification import ClassificationMethods

warnings.filterwarnings('ignore')

__author__ = "Nicolas de Montigny"

__all__ = ['bacteria_classification_train_cv']

TRAINING_DATASET_NAME = 'train'
VALIDATION_DATASET_NAME = 'validation'
TEST_DATASET_NAME = 'test'

# Initialisation / validation of parameters from CLI
################################################################################
def bacteria_classification_train_cv(opt):

    # Verify that model type is valid / choose default
    if opt['model_type'] is None:
        opt['model_type'] = 'cnn'

    # Validate training parameters
    verify_positive_int(opt['batch_size'], 'batch_size')
    verify_positive_int(opt['training_epochs'], 'number of iterations in neural networks training')
    
    outdirs = define_create_outdirs(opt['outdir'])
    
    # Initialize cluster
    init_ray_cluster(opt['workdir'])

# Data loading
################################################################################

    db_data, db_ds = verify_load_db(opt['data_bacteria'])

    # Validate and extract list of taxas
    if opt['taxa'] is not None:
        lst_taxas = verify_taxas(opt['taxa'], db_data['taxas'])
    else:
        lst_taxas = db_data['taxas'].copy()
        
    if 'domain' in lst_taxas:
        lst_taxas.remove('domain')
    
    # Verify need for scaling
    scaling = verify_need_scaling(db_data)
    
    for taxa in lst_taxas:

        if opt['test'] is not None:
            test_data, test_ds = verify_load_metagenome(opt['test'])
        else:
            test_data, test_ds = split_sim_dataset(db_ds, db_data, f"{TEST_DATASET_NAME}_{opt['database_name']}")
        if opt['validation'] is not None:
            val_data, val_ds = verify_load_metagenome(opt['validation'])
        else:
            val_data, val_ds = split_sim_dataset(db_ds, db_data, f"{VALIDATION_DATASET_NAME}_{opt['database_name']}")

        datasets = {
            TRAINING_DATASET_NAME : db_ds,
            TEST_DATASET_NAME : test_ds,
            VALIDATION_DATASET_NAME : val_ds
        }

# Training and cross-validation of models for classification of bacterias
################################################################################

        clf = ClassificationMethods(
            db_data = db_data,
            outdirs = outdirs,
            db_name = opt['database_name'],
            clf_multiclass = opt['model_type'],
            taxa = taxa,
            batch_size = opt['batch_size'],
            training_epochs = opt['training_epochs'],
            scaling = scaling
        )

        t_s = time()

        cv_scores = clf.cross_validation(datasets)

        t_clf = time() - t_s

        print(f"Caribou finished training and cross-validating the {opt['model_type']} model at taxa {taxa} in {t_clf} seconds")

# Argument parsing from CLI
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script trains and cross-validates a model for the bacteria classification step.')
    # Database
    parser.add_argument('-db','--data_bacteria', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the bacteria database')
    parser.add_argument('-dn','--database_name', required=True, help='Name of the bacteria database used to name files')
    # Optional datasets
    parser.add_argument('-v','--validation', default=None, type=Path, help='PATH to a npz file containing the k-mers profile for the validation dataset')
    parser.add_argument('-t','--test', default=None, type=Path, help='PATH to a npz file containing the k-mers profile for the test dataset')
    # Parameters
    parser.add_argument('-model','--model_type', default='lstm_attention', choices=['sgd','mnb','lstm_attention','cnn','widecnn'], help='The type of model to train')
    parser.add_argument('-tx','--taxa', default=None, help='The taxonomic level to use for the classification, defaults to None. Can be one level or a list of levels separated by commas.')
    parser.add_argument('-bs','--batch_size', default=32, type=int, help='Size of the batch size to use, defaults to 32')
    parser.add_argument('-e','--training_epochs', default=100, type=int, help='The number of training iterations for the neural networks models if one ise chosen, defaults to 100')
    parser.add_argument('-o','--outdir', required=True, type=Path, help='PATH to a directory on file where outputs will be saved')
    parser.add_argument('-wd','--workdir', default='/tmp/spill', type=Path, help='Optional. Path to a working directory where Ray Tune will output and spill tuning data')
    args = parser.parse_args()

    opt = vars(args)

    bacteria_classification_train_cv(opt)
