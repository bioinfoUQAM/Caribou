#!/usr/bin python3

import ray
import os.path
import argparse

from utils import *
from pathlib import Path
from time import time
from models.classification import ClassificationMethods

__author__ = "Nicolas de Montigny"

__all__ = ['bacteria_extraction_train_cv']

# Initialisation / validation of parameters from CLI
################################################################################
def bacteria_extraction(opt):
    # Verify existence of files and load data
    data_bacteria = verify_load_data(opt['data_bacteria'])
    if opt['data_host'] is not None:
        data_host = verify_load_data(opt['data_host'])
        verify_concordance_klength(len(data_bacteria['kmers'][0]), len(data_host['kmers'][0]))
    data_metagenome = verify_load_data(opt['data_metagenome'])

    k_length = len(data_bacteria['kmers'][0])

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
    ray.init()
    
# Definition of model for bacteria extraction / host removal + execution
################################################################################
    if opt['host_name'] is None:
        clf = ClassificationMethods(
            database_k_mers = data_bacteria,
            k = k_length,
            outdirs = outdirs,
            database = opt['database_name'],
            classifier_binary = opt['model_type'],
            taxa = 'domain',
            batch_size = opt['batch_size'],
            training_epochs = opt['training_epochs'],
            verbose = opt['verbose'],
            cv = False
        )
    else:
        clf = ClassificationMethods(
            database_k_mers = (data_bacteria, data_host),
            k = k_length,
            outdirs = outdirs,
            database = opt['database_name'],
            classifier_binary = opt['model_type'],
            taxa = 'domain',
            batch_size = opt['batch_size'],
            training_epochs = opt['training_epochs'],
            verbose = opt['verbose'],
            cv = False
        )
    t_start = time()
    clf.execute_training()
    t_end = time()
    t_train = t_end - t_start

# Execution of bacteria extraction / host removal on metagenome + save results
################################################################################
    def populate_save_data(clf):
        clf_data = {
            'sequence': clf.classified_data['sequence'].copy(),
            'profile' : clf.classified_data['domain']['bacteria'],
            'kmers' : data_metagenome['kmers'],
            'ids' : clf.classified_data['domain']['bacteria_ids'],
            'unknown_profile' : clf.classified_data['domain']['unknown'],
            'unknown_ids' : clf.classified_data['domain']['unknown_ids'],
        }
        if 'host' in clf.classified_data.keys():
            clf_data['host_profile'] = clf.classified_data['host']['classification']
            clf_data['host_ids'] = clf.classified_data['host']['classified_ids']
        clf_file = os.path.join(outdirs['results_dir'], opt['metagenome_name'] + '_extracted.npz')
        save_Xy_data(clf_data, clf_file)
        
    t_start = time()
    end_taxa = clf.execute_classification(data_metagenome)
    t_end = time()
    t_classify = t_end - t_start

    if end_taxa is None:
        populate_save_data(clf)
        print(f"Caribou finished training the {opt['model_type']} model and extracting bacteria with it. \
            \nThe training step took {t_train} seconds and the classification step took {t_classify} seconds.")
    else:
        print(f"Caribou finished training the {opt['model_type']} model but there was no data to classify. \
            \nThe training step took {t_train} seconds and the classification step took {t_classify} seconds.")

# Argument parsing from CLI
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script trains a model and extracts bacteria / host sequences.')
    parser.add_argument('-db','--data_bacteria', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the bacteria database')
    parser.add_argument('-dh','--data_host', default=None, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the host')
    parser.add_argument('-mg','--data_metagenome', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the metagenome to classify')
    parser.add_argument('-dt','--database_name', required=True, help='Name of the bacteria database used to name files')
    parser.add_argument('-ds','--host_name', default=None, help='Name of the host database used to name files')
    parser.add_argument('-mn','--metagenome_name', required=True, help='Name of the metagenome to classify used to name files')
    parser.add_argument('-model','--model_type', default=None, choices=[None,'onesvm','linearsvm','attention','lstm','deeplstm'], help='The type of model to train')
    parser.add_argument('-bs','--batch_size', default=32, type=int, help='Size of the batch size to use, defaults to 32')
    parser.add_argument('-e','--training_epochs', default=100, type=int, help='The number of training iterations for the neural networks models if one ise chosen, defaults to 100')
    parser.add_argument('-v','--verbose', action='store_true', help='Should the program be verbose')
    parser.add_argument('-o','--outdir', required=True, type=Path, help='PATH to a directory on file where outputs will be saved')
    parser.add_argument('-wd','--workdir', default=None, type=Path, help='Optional. Path to a working directory where Ray Tune will output and spill tuning data')
    args = parser.parse_args()

    opt = vars(args)

    bacteria_extraction(opt)