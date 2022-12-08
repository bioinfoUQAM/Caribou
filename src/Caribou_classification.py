#!/usr/bin python3

import ray
import logging
import argparse

from utils import *
from pathlib import Path
from models.classification import ClassificationMethods

__author__ = "Nicolas de Montigny"

__all__ = ['bacteria_classification_train_cv']

# Suppress Tensorflow warnings
################################################################################
logging.set_verbosity(logging.ERROR)

# Initialisation / validation of parameters from CLI
################################################################################
def bacteria_classification(opt):
    # Verify existence of files and load data
    verify_load_data(opt['data_bacteria'])
    data_metagenome = verify_load_data(opt['data_metagenome'])
    k_length = len(data_bacteria['kmers'][0])

    # Verify that model type is valid / choose default depending on host presence
    if opt['model_type'] is None:
        opt['model_type'] = 'cnn'

    # Validate training parameters
    verify_positive_int(opt['batch_size'], 'batch_size')
    verify_positive_int(opt['training_epochs'], 'number of iterations in neural networks training')
    
    outdirs = define_create_outdirs(opt['outdir'])

    # Validate and extract list of taxas
    list_taxas = verify_taxas(opt['taxa'], data_bacteria['taxas'])

    # Initialize cluster
    ray.init()

# Definition of model for bacteria taxonomic classification + training
################################################################################
    clf = ClassificationMethods(
        database_k_mers = data_bacteria,
        k = k_length,
        outdirs = outdirs,
        database = opt['database_name'],
        classifier_multiclass = opt['model_type'],
        taxa = list_taxas,
        batch_size = opt['batch_size'],
        training_epochs = opt['training_epochs'],
        verbose = opt['verbose'],
        cv = True
    )
    clf.execute_training()

# Execution of bacteria taxonomic classification on metagenome + save results
################################################################################
    clf.execute_classification(data_metagenome)

    clf_data = {'sequence' : clf.classified_data['sequence'].copy()}
    
    if 'domain' in clf_data['sequence'] and len(data_metagenome['classified_ids']) > 0:
        clf_data['domain']['profile'] = data_metagenome['profile']
        clf_data['domain']['kmers'] = data_metagenome['kmers']
        clf_data['domain']['ids'] = data_metagenome['ids']
        clf_data['domain']['classification'] = data_metagenome['classification']
        clf_data['domain']['classified_ids'] = data_metagenome['classified_ids']
        clf_data['domain']['unknown_profile'] = data_metagenome['unknown_profile']
        clf_data['domain']['unknown_ids'] = data_metagenome['unknown_ids']

    for taxa in clf_data['sequence']:
        clf_data[taxa]['profile'] = clf.classified_data[taxa]['unknown']
        clf_data[taxa]['kmers'] = data_metagenome['kmers']
        clf_data[taxa]['ids'] = clf.classified_data[taxa]['unknown_ids']
        clf_data[taxa]['classification'] = clf.classified_data['classification']
        clf_data[taxa]['classified_ids'] = clf.classified_data['classified_ids']
    
    clf_file = os.path.join(outdirs['results'], opt['metagenome_name'] + '_classified.npz')
    save_Xy_data(clf_data, clf_file)

    print("Caribou finished training the {} model and classifying bacterial sequences at {} taxonomic level with it".format(opt['model_type']))

# Argument parsing from CLI
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script trains and cross-validates a model for the bacteria classification step.')
    parser.add_argument('-db','--data_bacteria', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the bacteria database')
    parser.add_argument('-dt','--database_name', required=True, help='Name of the bacteria database used to name files')
    parser.add_argument('-mg','--data_metagenome', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the metagenome to classify')
    parser.add_argument('-mn','--metagenome_name', required=True, help='Name of the metagenome to classify used to name files')
    parser.add_argument('-model','--model_type', default='lstm_attention', choices=['sgd','mnb','lstm_attention','cnn','widecnn'], help='The type of model to train')
    parser.add_argument('-t','--taxa', default='species', help='The taxonomic level to use for the classification, defaults to species. Can be one level or a list of levels separated by commas.')
    parser.add_argument('-bs','--batch_size', default=32, type=int, help='Size of the batch size to use, defaults to 32')
    parser.add_argument('-e','--training_epochs', default=100, type=int, help='The number of training iterations for the neural networks models if one ise chosen, defaults to 100')
    parser.add_argument('-v','--verbose', action='store_true', help='Should the program be verbose')
    parser.add_argument('-o','--outdir', required=True, type=Path, help='PATH to a directory on file where outputs will be saved')
    parser.add_argument('-wd','--workdir', default='~/ray_results', type=Path, help='Optional. Path to a working directory where Ray Tune will output and spill tuning data')
    args = parser.parse_args()

    opt = vars(args)

    bacteria_classification(opt)
