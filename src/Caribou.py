#!/usr/bin python3
import os
import ray
import argparse
import configparser

from pathlib import Path

from data.build_data import build_load_save_data
from models.classification import ClassificationMethods
from outputs.out import Outputs

from tensorflow.compat.v1 import logging


__author__ = 'Nicolas de Montigny'

__all__ = ['caribou']

# Suppress Tensorflow warnings
################################################################################
logging.set_verbosity(logging.ERROR)

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
        if not os.path.isfile(file):
            raise ValueError('Cannot find file {} ! Exiting\n'.format(file))

    if host not in ['none', 'None', None]:
        for file in [host_seq_file, host_cls_file]:
            if not os.path.isfile(file):
                raise ValueError('Cannot find file {} ! Exiting\n'.format(file))

    # Verify path for saving
    outdir_path, outdir_folder = os.path.split(outdir)
    if not os.path.isdir(outdir_folder) and os.path.exists(outdir_path):
        print("Created output folder")
        os.makedirs(outdir)
    elif not os.path.exists(outdir_path):
        raise ValueError("Cannot find where to create output folder ! Exiting")

    # settings
    if type(k_length) != int or k_length <= 0:
        raise ValueError(
            'Invalid kmers length ! Please enter a positive integer ! Exiting\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')
    if binary_classifier not in ['onesvm','linearsvm','attention','lstm','deeplstm']:
        raise ValueError(
            'Invalid host extraction classifier ! Exiting\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')
    if multi_classifier not in ['ridge','svm','mlr','mnb','lstm_attention','cnn','widecnn']:
        raise ValueError(
            'Invalid multiclass bacterial classifier ! Exiting\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')
    if cv not in [True, False, None]:
        raise ValueError(
            'Invalid value for cross_validation ! Please use boolean values ! Exiting\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')
    if type(n_cvJobs) != int or n_cvJobs <= 0:
        raise ValueError(
            'Invalid number of cross validation jobs ! Please enter a positive integer ! Exiting\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')
    if verbose not in [True, False, None]:
        raise ValueError(
            'Invalid value for verbose parameter ! Please use boolean values ! Exiting\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')
    if type(training_batch_size) != int or training_batch_size <= 0:
        raise ValueError(
            'Invalid number of training batch size ! Please enter a positive integer ! Exiting\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')
    if training_epochs <= 0:
        raise ValueError(
            'Invalid number of iterations for training neural networks ! Please enter a value bigger than 0 ! Exiting\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')
    if not 0 < classif_threshold <= 1 or type(classif_threshold) != float:
        raise ValueError(
            'Invalid confidence threshold for classifying bacterial sequences ! Please enter a value between 0 and 1 ! Exiting\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')

    # outputs
    if abundance_stats not in [True, False, None]:
        raise ValueError(
            'Invalid value for output in abundance table form ! Please use boolean values ! Exiting\n' + 
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')
    if kronagram not in [True, False, None]:
        raise ValueError(
            'Invalid value for output in Kronagram form ! Please use boolean values ! Exiting\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')
    if full_report not in [True, False, None]:
        raise ValueError(
            'Invalid value for output in full report form ! Please use boolean values ! Exiting\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')
    if extract_fasta not in [True, False, None]:
        raise ValueError(
            'Invalid value for output in fasta extraction form ! Please use boolean values ! Exiting\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')

    # Adjust classifier based on host presence or not
    if host in ['none', 'None', None]:
        binary_classifier = 'onesvm'

    # Check batch_size
    if multi_classifier in ['cnn','widecnn'] and training_batch_size < 20:
        training_batch_size = 20

    # Folders creation for output
    outdirs = {}
    outdirs['main_outdir'] = outdir
    outdirs['data_dir'] = os.path.join(outdirs['main_outdir'], 'data/')
    outdirs['models_dir'] = os.path.join(outdirs['main_outdir'], 'models/')
    outdirs['results_dir'] = os.path.join(outdirs['main_outdir'], 'results/')
    os.makedirs(outdirs['main_outdir'], mode=0o700, exist_ok=True)
    os.makedirs(outdirs['data_dir'], mode=0o700, exist_ok=True)
    os.makedirs(outdirs['models_dir'], mode=0o700, exist_ok=True)
    os.makedirs(outdirs['results_dir'], mode=0o700, exist_ok=True)
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
