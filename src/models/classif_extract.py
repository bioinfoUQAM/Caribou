
import os
import sys
import ray
import pickle
import cloudpickle

import pandas as pd

from models.ray_sklearn import SklearnModel
from models.ray_keras_tf import KerasTFModel
from utils import load_Xy_data, save_Xy_data

from models.classif_utils import ClassificationUtils

__author__ = 'Nicolas de Montigny'

__all__ = ['ClassificationExtraction']

class ClassificationExtraction(ClassificationUtils):
    """
    Class for extracting bacterial sequences from metagenomes using a trained model

    ----------
    Attributes
    ----------

    ----------
    Methods
    ----------

    """
    def __init__(
        self,
        metagenome_k_mers,
        database_k_mers,
        k,
        outdirs,
        dataset,
        training_epochs = 100,
        classifier = 'deeplstm',
        batch_size = 32,
        verbose = True,
        cv = True
    ):
        super().__init__(
            self,
            database_k_mers,
            k,
            outdirs,
            dataset,
            training_epochs,
            classifier,
            batch_size,
            verbose,
            cv
        )
        # Parameters
        self.metagenome_k_mers = metagenome_k_mers
        # Empty initializations
        # classified_data is a dictionnary containing data dictionnaries at each classified level:
        self.model = None
        self.model_file = '{}{}_{}.pkl'.format(outdirs['models_dir'], classifier, 'domain')
        self.classified_data = {
            'order': ['bacteria', 'host', 'unclassified'],
            'bacteria' : {},
            'host' : {},
            'unclassified' : {}
        }

        def extract_bacteria(self):
