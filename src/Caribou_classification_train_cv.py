#!/usr/bin python3

import ray
import argparse

from utils import *
from pathlib import Path
from models.classification import ClassificationMethods

__author__ = "Nicolas de Montigny"

__all__ = ['bacteria_classification_train_cv']

# Initialisation / validation of parameters from CLI
################################################################################
def bacteria_classification_train_cv(opt):
    # Verify existence of files and load data
    data_bacteria = verify_load_data(opt['data_bacteria'])
    k_length = len(data_bacteria['kmers'][0])

    # Verify that model type is valid / choose default depending on host presence
    if opt['model_type'] is None:
        opt['model_type'] = 'cnn'

    # Validate training parameters
    verify_positive_int(opt['batch_size'], 'batch_size')
    verify_positive_int(opt['training_epochs'], 'number of iterations in neural networks training')
    
    outdirs = define_create_outdirs(opt['outdir'])
    
    lst_taxas = data_bacteria['taxas'].copy()
    lst_taxas.remove('domain')

    # Initialize cluster
    ray.init()

# Training and cross-validation of models for classification of bacterias
################################################################################
    ClassificationMethods(
        database_k_mers = data_bacteria,
        k = k_length,
        outdirs = outdirs,
        database = opt['database_name'],
        classifier_binary = None,
        classifier_multiclass = opt['model_type'],
        taxa = lst_taxas,
        batch_size = opt['batch_size'],
        training_epochs = opt['training_epochs'],
        verbose = opt['verbose'],
        cv = True
    ).execute_training()

    print("Caribou finished training and cross-validating the {} model".format(opt['model_type']))

# Argument parsing from CLI
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script trains and cross-validates a model for the bacteria classification step.')
    parser.add_argument('-db','--data_bacteria', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the bacteria database')
    parser.add_argument('-dt','--database_name', required=True, help='Name of the bacteria database used to name files')
    parser.add_argument('-model','--model_type', default='lstm_attention', choices=['sgd','mnb','lstm_attention','cnn','widecnn'], help='The type of model to train')
    parser.add_argument('-bs','--batch_size', default=32, type=int, help='Size of the batch size to use, defaults to 32')
    parser.add_argument('-e','--training_epochs', default=100, type=int, help='The number of training iterations for the neural networks models if one ise chosen, defaults to 100')
    parser.add_argument('-v','--verbose', action='store_true', help='Should the program be verbose')
    parser.add_argument('-o','--outdir', required=True, type=Path, help='PATH to a directory on file where outputs will be saved')
    parser.add_argument('-wd','--workdir', default=None, type=Path, help='Optional. Path to a working directory where Ray Tune will output and spill tuning data')
    args = parser.parse_args()

    opt = vars(args)

    bacteria_classification_train_cv(opt)
    """
    Traceback (most recent call last):
    File "/usr/local/bin/Caribou_classification_train_cv.py", line 71, in <module>
        bacteria_classification_train_cv(opt)
    File "/usr/local/bin/Caribou_classification_train_cv.py", line 39, in bacteria_classification_train_cv
        ClassificationMethods(
    File "/usr/local/lib/python3.8/dist-packages/models/classification.py", line 127, in execute_training
        self._train_model(taxa)
    File "/usr/local/lib/python3.8/dist-packages/models/classification.py", line 135, in _train_model
        self._multiclass_training(taxa)
    File "/usr/local/lib/python3.8/dist-packages/models/classification.py", line 233, in _multiclass_training
        taxa: pd.DataFrame(
    File "/usr/local/lib/python3.8/dist-packages/pandas/core/frame.py", line 672, in __init__
        mgr = ndarray_to_mgr(
    File "/usr/local/lib/python3.8/dist-packages/pandas/core/internals/construction.py", line 324, in ndarray_to_mgr
        _check_values_indices_shape_match(values, index, columns)
    File "/usr/local/lib/python3.8/dist-packages/pandas/core/internals/construction.py", line 393, in _check_values_indices_shape_match
        raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}")
    ValueError: Shape of passed values is (7441, 7), indices imply (7441, 6)
    """