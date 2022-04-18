#!/usr/bin python3

from Caribou.models.bacteria_extraction import bacteria_extraction
from Caribou.utils import load_Xy_data

import pandas as pd

from tensorflow.compat.v1 import ConfigProto, Session
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.config import list_physical_devices

import sys
import os.path
import argparse

from os import makedirs
from pathlib import Path

__author__ = "Nicolas de Montigny"

__all__ = ['bacteria_extraction_train_cv']

# GPU & CPU setup
################################################################################
gpus = list_physical_devices('GPU')
if gpus:
    config = ConfigProto(device_count={'GPU': len(gpus), 'CPU': os.cpu_count()})
    sess = Session(config=config)
    set_session(sess);

# Initialisation / validation of parameters from CLI
################################################################################
def bacteria_extraction_train_cv(opt):

    # Verify existence of files and load data
    if not os.path.isfile(opt['data_bacteria']):
        print("Cannot find file {} ! Exiting".format(file))
        sys.exit()
    else:
        data_bacteria = load_Xy_data(opt['data_bacteria'])
        # Infer k-mers length from the extracted bacteria profile
        k_length = len(data_bacteria['kmers_list'][0])
        # Verify that kmers profile file exists
        if not os.path.isfile(data_bacteria['X']):
            print("Cannot find file {} ! Exiting".format(os.path.isfile(data_bacteria['X'])))
            sys.exit()

    # Verify existence of files for host data + concordance and load
    if opt['data_host'] is not None:
        if not os.path.isfile(opt['data_host']):
            print("Cannot find file {} ! Exiting".format(file))
            sys.exit()
        else:
            data_host = load_Xy_data(opt['data_host'])
            # Verify concordance of k length between datasets
            if k_length != len(data_host['kmers_list'][0]):
                print("K length of bacteria dataset is {} while K length from host is {}").format(k_length, len(data_host['kmers_list'][0]))
                print("K length between datasets is inconsistent ! Exiting")
                sys.exit()
            else:
                # Verify that kmers profile file exists
                if not os.path.isfile(data_host['X']):
                    print("Cannot find file {} ! Exiting".format(os.path.isfile(data_host['X'])))
                    sys.exit()
    else:
        host = None

    # Verify that model type is valid / choose default depending on host presence
    if host is None:
        opt['model_type'] = 'onesvm'
    elif opt['model_type'] is None and host is not None:
        opt['model_type'] = 'attention'

    # Validate batch size
    if opt['batch_size'] <= 0:
        print("Invalid batch size ! Exiting")
        sys.exit()

    # Validate number of cv jobs
    if opt['nb_cv_jobs'] <= 0:
        print("Invalid number of cross-validation jobs")
        sys.exit()

    # Validate number of epochs
    if opt['training_epochs'] <= 0:
        print("Invalid number of training iterations for neural networks")
        sys.exit()

    # Validate path for saving
    outdir_path, outdir_folder = os.path.split(opt['outdir'])
    if not os.path.isdir(outdir) and os.path.exists(outdir_path):
        print("Created output folder")
        os.makedirs(outdir)
    elif not os.path.exists(outdir_path):
        print("Cannot find where to create output folder ! Exiting")
        sys.exit()

    # Folders creation for output
    outdirs = {}
    outdirs["main_outdir"] = outdir
    outdirs["data_dir"] = os.path.join(outdirs["main_outdir"], "data/")
    outdirs["models_dir"] = os.path.join(outdirs["main_outdir"], "models/")
    makedirs(outdirs["main_outdir"], mode=0o700, exist_ok=True)
    makedirs(outdirs["models_dir"], mode=0o700, exist_ok=True)

# Training and cross-validation of models for bacteria extraction / host removal
################################################################################

    if host is None:
        bacteria_extraction(None,
            data_bacteria,
            k_length,
            outdirs,
            opt['database_name'],
            opt['training_epochs'],
            classifier = opt['model_type'],
            batch_size = opt['batch_size'],
            verbose = opt['verbose'],
            cv = True,
            n_jobs = opt['nb_cv_jobs']
            )
    else:
        bacteria_extraction(None,
            (data_bacteria, data_host),
            k_length,
            outdirs,
            opt['database_name'],
            opt['training_epochs'],
            classifier = opt['model_type'],
            batch_size = opt['batch_size'],
            verbose = opt['verbose'],
            cv = True,
            n_jobs = opt['nb_cv_jobs']
            )

    print("Caribou finished training and cross-validating the {} model without faults").format(opt['model_type'])


# Argument parsing from CLI
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script trains and cross-validates a model for the bacteria extraction / host removal step.')
    parser.add_argument('-db','--data_bacteria', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the bacteria database')
    parser.add_argument('-dh','--data_host', default=None, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the host')
    parser.add_argument('-dt','--database_name', required=True, help='Name of the bacteria database used to name files')
    parser.add_argument('-model','--model_type', default=None, choices=[None,'linearsvm','attention','lstm','deeplstm'], help='The type of model to train')
    parser.add_argument('-bs','--batch_size', default=32, type=int, help='Size of the batch size to use, defaults to 32')
    parser.add_argument('-cv','--nb_cv_jobs', default=10, type=int, help='The number of cross validation jobs to run, defaults to 10')
    parser.add_argument('-e','--training_epochs', default=100, type=int, help='The number of training iterations for the neural networks models if one ise chosen, defaults to 100')
    parser.add_argument('-v','--verbose', action='store_true', help='Should the program be verbose')
    parser.add_argument('-o','--outdir', required=True, type=Path, help='PATH to a directory on file where outputs will be saved')
    args = parser.parse_args()

    opt = vars(args)

    bacteria_extraction_train_cv(opt)
