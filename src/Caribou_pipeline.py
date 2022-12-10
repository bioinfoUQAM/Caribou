#!/usr/bin python3

import ray
import argparse
import configparser

from utils import *
from pathlib import Path
from outputs.out import Outputs
from data.build_data import build_load_save_data
from models.classification import ClassificationMethods

__author__ = 'Nicolas de Montigny'

__all__ = ['caribou']

# Part 0 - Initialisation / extraction of parameters from config file
################################################################################
def caribou(opt):
    # Get argument values from config file
    config_file = opt['config']
    config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())

    with open(config_file, 'r') as cf:
        config.read_file(cf)

    # names
    database = config.get('name', 'database', fallback = 'database')
    metagenome = config.get('name', 'metagenome', fallback = 'metagenome')
    host = config.get('name', 'host', fallback = None)

    # io
    database_seq_file = config.get('io', 'database_seq_file')
    database_cls_file = config.get('io', 'database_cls_file')
    host_seq_file = config.get('io', 'host_seq_file', fallback = None)
    host_cls_file = config.get('io', 'host_cls_file', fallback = None)
    metagenome_seq_file = config.get('io', 'metagenome_seq_file')
    outdir = config.get('io', 'outdir')

    # settings
    k_length = config.getint('settings', 'k', fallback = 35)
    binary_classifier = config.get('settings', 'host_extractor', fallback = 'attention')
    multi_classifier = config.get('settings', 'bacteria_classifier', fallback = 'lstm_attention')
    cv = config.getboolean('settings', 'cross_validation', fallback = True)
    n_cvJobs = config.getint('settings', 'nb_cv_jobs', fallback = 1)
    verbose = config.getboolean('settings', 'verbose', fallback = True)
    training_batch_size = config.getint('settings', 'training_batch_size', fallback = 32)
    training_epochs = config.getint('settings','neural_network_training_iterations', fallback = 100)
    classif_threshold = config.getfloat('settings', 'classification_threshold', fallback = 0.8)

    # outputs
    abundance_stats = config.getboolean('outputs', 'abundance_report', fallback = True)
    kronagram = config.getboolean('outputs', 'kronagram', fallback = True)
    full_report = config.getboolean('outputs', 'full_report', fallback = True)
    extract_fasta = config.getboolean('outputs', 'extract_fasta', fallback = True)

# Part 0.5 - Validation of parameters and environment
################################################################################

    # io
    for file in [database_seq_file, database_cls_file, metagenome_seq_file]:
        verify_file(file)

    host = verify_host(host)
    if host is not None:
        for file in [host_seq_file, host_cls_file]:
            file_exist(file)
    else:
        # Adjust classifier based on host presence or not
        binary_classifier = 'onesvm'

    # settings
    verify_positive_int(k_length, 'kmers length')
    verify_binary_classifier(binary_classifier)
    verify_multiclass_classifier(multi_classifier)
    verify_boolean(cv, 'cross validation')
    verify_boolean(verbose, 'verbose parameter')
    verify_positive_int(training_batch_size, 'training batch size')
    verify_positive_int(training_epochs, 'number of iterations in neural networks training')
    verify_0_1(classif_threshold, 'classification threshold')

    # outputs
    verify_boolean(abundance_stats, 'output in abundance table form')
    verify_boolean(kronagram, 'output in Kronagram form')
    verify_boolean(full_report, 'output in full report form')
    verify_boolean(extract_fasta, 'output in fasta extraction form')
    
    # Check batch_size
    if multi_classifier in ['cnn','widecnn'] and training_batch_size < 20:
        training_batch_size = 20

    # Folders creation for output
    outdirs = define_create_outdirs(outdir)
    
    # Initialize cluster
    ray.init()

# Part 1 - K-mers profile extraction
################################################################################

    if host is not None:
        # Reference Database and Host
        k_profile_database, k_profile_host = build_load_save_data((database_seq_file, database_cls_file),
            (host_seq_file, host_cls_file),
            outdirs['data_dir'],
            database,
            host,
            k = k_length,
        )
    else:
        # Reference Database Only
        k_profile_database = build_load_save_data((database_seq_file, database_cls_file),
            host,
            outdirs['data_dir'],
            database,
            host,
            k = k_length,
        )

    # Metagenome to analyse
    k_profile_metagenome = build_load_save_data(metagenome_seq_file,
        None,
        outdirs['data_dir'],
        metagenome,
        host,
        kmers_list = k_profile_database['kmers']
    )

# Part 2 - Instanciation of the classifiers
################################################################################

    if host is None:
        recursive_classifier = ClassificationMethods(
            database_k_mers = k_profile_database,
            k = k_length,
            outdirs = outdirs,
            database = database,
            classifier_binary = binary_classifier,
            classifier_multiclass = multi_classifier,
            taxa = None,
            threshold = classif_threshold,
            batch_size = training_batch_size,
            training_epochs = training_epochs,
            verbose = verbose,
            cv = cv
        )
    else:
        recursive_classifier = ClassificationMethods(
            database_k_mers = (k_profile_database, k_profile_host),
            k = k_length,
            outdirs = outdirs,
            database = database,
            classifier_binary = binary_classifier,
            classifier_multiclass = multi_classifier,
            taxa = None,
            threshold = classif_threshold,
            batch_size = training_batch_size,
            training_epochs = training_epochs,
            verbose = verbose,
            cv = cv
        )

# Part 3 - Recursive classification based on the database data
################################################################################

    # Train the models
    recursive_classifier.execute_training()

    # Classify the data from the metagenome
    recursive_classifier.execute_classication(k_profile_metagenome)

    # Get classification results dictionnary
    classified_data = recursive_classifier.classified_data

# Part 4 - Outputs for biological analysis of bacterial population
################################################################################

    outputs = Outputs(k_profile_database,
                      outdirs['results_dir'],
                      k_length,
                      multi_classifier,
                      metagenome,
                      host,
                      classified_data)

    # Output desired files according to parameters
    if abundance_stats is True:
        outputs.abundances()
    if kronagram is True:
        outputs.kronagram()
    if full_report is True:
        outputs.report()
    if extract_fasta is True:
        outputs.fasta()

    print('Caribou finished executing without faults and all results were outputed in the designated folders')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script runs the entire Caribou analysis Pipeline')
    parser.add_argument('-c','--config', required=True, type=Path, help='PATH to a configuration file containing the choices made by the user. Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')
    args = parser.parse_args()

    opt = vars(args)

    caribou(opt)
