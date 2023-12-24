#!/usr/bin python3

import os
import argparse

from utils import *
from time import time
from pathlib import Path
from models.reads_simulation import split_sim_dataset
from models.classification import ClassificationMethods

__author__ = "Nicolas de Montigny"

__all__ = ['bacteria_classification_train_cv']

TRAINING_DATASET_NAME = 'train'
VALIDATION_DATASET_NAME = 'validation'

# Initialisation / validation of parameters from CLI
################################################################################
def bacteria_classification(opt):
    
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

    if opt['validation'] is not None:
        val_data, val_ds = verify_load_metagenome(opt['validation'])
    else:
        val_data, val_ds = split_sim_dataset(db_ds, db_data, f"{VALIDATION_DATASET_NAME}_{opt['database_name']}")

    datasets = {
        TRAINING_DATASET_NAME : db_ds,
        VALIDATION_DATASET_NAME : val_ds
    }

    metagenome_data, metagenome_ds = verify_load_metagenome(opt['data_metagenome'])

# Definition of model for bacteria taxonomic classification
################################################################################
    
    clf = ClassificationMethods(
        db_data = db_data,
        outdirs = outdirs,
        db_name = opt['database_name'],
        clf_multiclass = opt['model_type'],
        taxa = 'domain',
        batch_size = opt['batch_size'],
        training_epochs = opt['training_epochs'],
        scaling = scaling
    )
    
# Execution of bacteria taxonomic classification on metagenome + save results
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
          Classification of bacteria from {opt['metagenome_name']} dataset was then executed in {t_clf} seconds.
          """)

# Argument parsing from CLI
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script trains a model and classifies bacteria sequences iteratively over known taxonomic levels.')
    # Database
    parser.add_argument('-db','--data_bacteria', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the bacteria database')
    parser.add_argument('-dt','--database_name', required=True, help='Name of the bacteria database used to name files')
    # Dataset
    parser.add_argument('-mg','--data_metagenome', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the metagenome to classify')
    parser.add_argument('-mn','--metagenome_name', required=True, help='Name of the metagenome to classify used to name files')
    # Optional datasets
    parser.add_argument('-v','--validation', default=None, type=Path, help='PATH to a npz file containing the k-mers profile for the validation dataset')
    # Parameters
    parser.add_argument('-model','--model_type', default='sgd', choices=['sgd','mnb','lstm_attention','cnn','widecnn'], help='The type of model to train')
    parser.add_argument('-tx','--taxa', default=None, help='The taxonomic level to use for the classification, defaults to species. Can be one level or a list of levels separated by commas.')
    parser.add_argument('-bs','--batch_size', default=32, type=int, help='Size of the batch size to use, defaults to 32')
    parser.add_argument('-e','--training_epochs', default=100, type=int, help='The number of training iterations for the neural networks models if one ise chosen, defaults to 100')
    parser.add_argument('-o','--outdir', required=True, type=Path, help='PATH to a directory on file where outputs will be saved')
    parser.add_argument('-wd','--workdir', default='/tmp/spill', type=Path, help='Optional. Path to a working directory where Ray Tune will output and spill tuning data')
    args = parser.parse_args()

    opt = vars(args)

    bacteria_classification(opt)
