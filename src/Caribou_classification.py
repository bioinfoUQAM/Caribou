#!/usr/bin python3

import os
import argparse

from utils import *
from time import time
from pathlib import Path
from models.classification import ClassificationMethods

__author__ = "Nicolas de Montigny"

__all__ = ['bacteria_classification_train_cv']

# Initialisation / validation of parameters from CLI
################################################################################
def bacteria_classification(opt):
    # Verify existence of files and load data
    data_bacteria = verify_load_data(opt['data_bacteria'])
    data_metagenome = verify_load_data(opt['data_metagenome'])
    k_length = len(data_bacteria['kmers'][0])
    
    if opt['preclassified_data'] is not None:
        preclassified_data = verify_load_preclassified(opt['preclassified_data'])
    else:
        preclassified_data = None

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

# Definition of model for bacteria taxonomic classification + training
################################################################################
    clf = ClassificationMethods(
        database_k_mers = data_bacteria,
        k = k_length,
        outdirs = outdirs,
        database = opt['database_name'],
        classifier_multiclass = opt['model_type'],
        taxa = lst_taxas,
        batch_size = opt['batch_size'],
        training_epochs = opt['training_epochs'],
        verbose = opt['verbose'],
        cv = False
    )
    
# Execution of bacteria taxonomic classification on metagenome + save results
################################################################################
    
    t_start = time()
    end_taxa = clf.execute_training_prediction(data_metagenome)
    t_end = time()
    t_classif = t_end - t_start
    clf_data = merge_save_data(
        clf.classified_data,
        data_metagenome,
        end_taxa,
        outdirs['results_dir'],
        opt['metagenome_name'],
        preclassified = preclassified_data,
    )
    if opt['taxa'] is None:
        opt['taxa'] = 'all'
    clf_data['classification'].to_csv(os.path.join(outdirs['results_dir'], f"classification_K{k_length}_{opt['taxa']}_{opt['model_type']}.csv"))
    if end_taxa is None:
        print(f"Caribou finished training the {opt['model_type']} model and classifying bacterial sequences at {opt['taxa']} taxonomic level with it. \
            \nThe training and classification steps took {t_classif} seconds to execute.")
    else:
        print(f"Caribou finished training the {opt['model_type']} model and classifying bacterial sequences at {opt['taxa']} taxonomic level until {end_taxa} because there were no more sequences to classify. \
            \nThe training and classification steps took {t_classif} seconds to execute.")

# Argument parsing from CLI
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script trains a model and classifies bacteria sequences iteratively over known taxonomic levels.')
    parser.add_argument('-db','--data_bacteria', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the bacteria database')
    parser.add_argument('-mg','--data_metagenome', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the metagenome to classify')
    parser.add_argument('-dt','--database_name', required=True, help='Name of the bacteria database used to name files')
    parser.add_argument('-mn','--metagenome_name', required=True, help='Name of the metagenome to classify used to name files')
    parser.add_argument('-pc','--preclassified_data', default=None, type=Path,help='Optional. PATH to a .npz file contianing classified data at another taxonomic level than the ones in the current analysis')
    parser.add_argument('-model','--model_type', default='lstm_attention', choices=['sgd','mnb','lstm_attention','cnn','widecnn'], help='The type of model to train')
    parser.add_argument('-t','--taxa', default=None, help='The taxonomic level to use for the classification, defaults to species. Can be one level or a list of levels separated by commas.')
    parser.add_argument('-bs','--batch_size', default=32, type=int, help='Size of the batch size to use, defaults to 32')
    parser.add_argument('-e','--training_epochs', default=100, type=int, help='The number of training iterations for the neural networks models if one ise chosen, defaults to 100')
    parser.add_argument('-v','--verbose', action='store_true', help='Should the program be verbose')
    parser.add_argument('-o','--outdir', required=True, type=Path, help='PATH to a directory on file where outputs will be saved')
    parser.add_argument('-wd','--workdir', default='/tmp/spill', type=Path, help='Optional. Path to a working directory where Ray Tune will output and spill tuning data')
    args = parser.parse_args()

    opt = vars(args)

    bacteria_classification(opt)
