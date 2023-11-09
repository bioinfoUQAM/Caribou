#!/usr/bin python3

import os
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

# Initialisation / validation of parameters from CLI
################################################################################
def bacteria_extraction(opt):

    # Verify that model type is valid / choose default depending on host presence
    if opt['host_name'] is None:
        opt['model_type'] = 'onesvm'
    elif opt['model_type'] is None and opt['host_name'] is not None:
        opt['model_type'] = 'attention'

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
            db_data, db_ds = verify_load_host_merge(opt['data_bacteria'], opt['data_host'])
            db_name = 'host_merged'
        else:
            db_data, db_ds = verify_load_db(opt['data_bacteria'])
            db_name = opt['dataset_name']

        val_ds, val_data = split_sim_dataset(db_ds, db_data, f'{VALIDATION_DATASET_NAME}_{db_name}')
    else:
        db_data, db_ds = verify_load_host_merge(opt['data_bacteria'], opt['data_host'])
        db_name = 'host_merged'

        val_ds, val_data = split_sim_dataset(db_ds, db_data, f'{VALIDATION_DATASET_NAME}_{db_name}')

        db_data, db_ds = verify_load_db(opt['data_bacteria'])
        db_name = opt['dataset_name']

    datasets = {
        TRAINING_DATASET_NAME : db_ds,
        VALIDATION_DATASET_NAME : val_ds
    }

    metagenome_data, metagenome_ds = verify_load_metagenome(opt['data_metagenome'])

# Definition of model for bacteria extraction / host removal
################################################################################
    
    clf = ClassificationMethods(
        db_data = db_data,
        outdirs = outdirs,
        db_name = opt['database_name'],
        clf_binary = opt['model_type'],
        taxa = 'domain',
        batch_size = opt['batch_size'],
        training_epochs = opt['training_epochs']
    )

# Execution of bacteria extraction / host removal on metagenome + save results
################################################################################
    
    t_s = time()
    clf.fit(datasets)
    t_fit = time() - t_s

    t_s = time()
    predictions = clf.predict(metagenome_ds)
    t_clf = time() - t_s

    Xy_file = os.path.join(outdirs['results_dir'], f"extracted_bacteria_{opt['metagenome_name']}_{opt['model_type']}.npz")
    save_Xy_data(predictions, Xy_file)

    print(f"""
          Caribou finished training the {opt['model_type']} model in {t_fit} seconds.
          Extraction of bacteria from {opt['metagenome_name']} dataset was then executed in {t_clf} seconds.
          """)

# Argument parsing from CLI
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script trains a model and extracts bacteria / host sequences.')
    # Database
    parser.add_argument('-db','--data_bacteria', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the bacteria database')
    parser.add_argument('-dh','--data_host', default=None, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the host')
    parser.add_argument('-dn','--database_name', required=True, help='Name of the bacteria database used to name files')
    parser.add_argument('-hn','--host_name', default=None, help='Name of the host database used to name files')
    # Dataset
    parser.add_argument('-dm','--data_metagenome', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the metagenome to classify')
    parser.add_argument('-mn','--metagenome_name', required=True, help='Name of the metagenome to classify used to name files')
    # Parameters
    parser.add_argument('-model','--model_type', default=None, choices=[None,'onesvm','linearsvm','attention','lstm','deeplstm'], help='The type of model to train')
    parser.add_argument('-bs','--batch_size', default=32, type=int, help='Size of the batch size to use, defaults to 32')
    parser.add_argument('-e','--training_epochs', default=100, type=int, help='The number of training iterations for the neural networks models if one ise chosen, defaults to 100')
    parser.add_argument('-o','--outdir', required=True, type=Path, help='PATH to a directory on file where outputs will be saved')
    parser.add_argument('-wd','--workdir', default='/tmp/spill', type=Path, help='Optional. Path to a working directory where Ray Tune will output and spill tuning data')
    args = parser.parse_args()

    opt = vars(args)

    bacteria_extraction(opt)