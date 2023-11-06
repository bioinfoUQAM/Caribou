import os
import ray
import cloudpickle

import numpy as np
import pandas as pd

from glob import glob
from typing import Dict
from shutil import rmtree
from utils import load_Xy_data
from models.sklearn.models import SklearnModel
from models.kerasTF.models import KerasTFModel

# Simulation class
from models.reads_simulation import readsSimulation

__author__ = 'Nicolas de Montigny'

__all__ = ['ClassificationMethods']

TRAINING_DATASET_NAME = 'train'
VALIDATION_DATASET_NAME = 'validation'
TEST_DATASET_NAME = 'test'

class ClassificationMethods():
    """
    Utilities class for classifying sequences from metagenomes using ray

    ----------
    Attributes
    ----------
    
    classified_data : dictionary
        Dictionary containing the classified data for each classified taxonomic level

    models : dictionary
        Dictionary containing the trained models for each taxonomic level

    ----------
    Methods
    ----------

    fit : function to call the fitting method
    
    predict : function to call the predicting method

    fit_predict : wrapper function for calling fit and predict

    cross_validation : function to call the cross-validation process
    
    """
    def __init__(
        self,
        database_k_mers: Dict,
        k: int,
        outdirs: Dict,
        database: str,
        classifier_binary: str = 'deeplstm',
        classifier_multiclass: str = 'widecnn',
        taxa: str = None,
        threshold: float = 0.8,
        batch_size: int = 32,
        training_epochs: int = 100,
        verbose: bool = True,
        cv: bool = False
    ):
        # Parameters
        self._k = k
        self._cv = cv
        self._taxas = taxa
        self._outdirs = outdirs
        self._database = database
        self._verbose = verbose
        self._threshold = threshold
        self._classifier_binary = classifier_binary
        self._classifier_multiclass = classifier_multiclass
        self._batch_size = batch_size
        self._training_epochs = training_epochs
        # Initialize with values
        self.classified_data = {
            'sequence': [],
            'classification' : None,
            'classified_ids' : [],
            'unknown_ids' : []
        }
        # Empty initializations
        self.models = {}
        self._host = False
        self._taxas_order = []
        self._host_data = None
        self._database_data = None
        self._training_datasets = None
        self._merged_training_datasets = None
        self._merged_database_host = None
        self.previous_taxa_unclassified = None
        # Extract database data 
        if isinstance(database_k_mers, tuple):
            self._host = True
            self._database_data = database_k_mers[0]
            self._host_data = database_k_mers[1]
        else:
            self._database_data = database_k_mers
        # Remove 'id' from kmers if present
        if 'id' in self._database_data['kmers']:
            self._database_data['kmers'].remove('id')
        if self._host and 'id' in self._host_data['kmers']:
            self._host_data['kmers'].remove('id')
        # Assign taxas order for top-down strategy
        self._taxas_order = self._database_data['taxas'].copy()
        self._taxas_order.reverse()
        # Automatic executions
        self._verify_assign_taxas(taxa)

    # Public functions
    #########################################################################################################
# TODO: Revise documentation in heading
# TODO: Remove parameters from global if they are only required for certain functions
# TODO: Finish transfering the functions & calls from the old version
# TODO: Validation of params before execution of private functions
    def fit(self, datasets, ):
        """
        Wrapper function to call the fitting method
        """
        # TODO: Pass training/validation data here

    def predict(self):
        """
        Wrapper function to call the predicting method
        """
        # TODO: Pass data to predict here

    def fit_predict(self):
        """
        Wrapper function for calling fit and predict
        """
        # TODO: Pass training/validation data here
        # TODO: Pass data to predict here
    
    def cross_validation(self):
        """
        Wrapper function to call the cross-validation method
        """
        # TODO: Pass training/validation data here
        # TODO: Pass testing data here

    # Private principal functions
    #########################################################################################################
# TODO: Pass training/validation data here
    def _fit(self):
        """
        Fit the given model to the training dataset
        """
        for taxa in self._taxas_order:
            if taxa in self._taxas:
                if taxa in ['domain','bacteria','host']:
                    clf = self._classifier_binary
                else:
                    clf = self._classifier_multiclass
                self._data_file = os.path.join(self._outdirs['data_dir'], f'Xy_{taxa}_database_K{self._k}_{clf}_{self._database}_data.npz')
                self._model_file = os.path.join(self._outdirs['models_dir'], f'{clf}_{taxa}.pkl')
                train = self._verify_load_data_model(self._data_file, self._model_file, taxa)
                if train:
                    if taxa in ['domain','bacteria','host']:
                        self._binary_training(taxa)
                    else:
                        self._multiclass_training(taxa)

# TODO: Pass data to predict here
    def _predict(self, data2classify):
        """
        Predict the given data using the trained model
        """
        files_lst = glob(os.path.join(data2classify['profile'],'*.parquet'))
        df = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))
        ids = data2classify['ids']
        if len(self.classified_data['sequence']) == 0:
            raise ValueError('Please train a model before executing classification')
        for i, taxa in enumerate(self.classified_data['sequence']):
            try:
                if i == 0:
                    df = self._classify_first(df, taxa, ids, data2classify['profile'])
                else:
                    df = self._classify_subsequent(df, taxa, ids, data2classify['profile'])
            except ValueError:
                print('Stopping classification prematurelly because there are no more sequences to classify')
                return taxa
        return None
    
    def _cross_validation(self):
        """
        Execute cross-validation of a model by fitting a model and predicting over a test dataset
        """

    # Private training secondary functions
    #########################################################################################################
# TODO: Remove data loading & verification from inside these functions
    def _binary_training(self, taxa):
        print('_binary_training')
        self._verify_classifier_binary()
        if self._classifier_binary == 'onesvm':
            self.models[taxa] = SklearnModel(
                self._classifier_binary,
                self._database,
                self._outdirs['models_dir'],
                self._outdirs['results_dir'],
                self._batch_size,
                self._training_epochs,
                self._k,
                taxa,
                self._database_data['kmers'],
                self._verbose
            )
        else:
            if self._classifier_binary == 'linearsvm':
                self.models[taxa] = SklearnModel(
                    self._classifier_binary,
                    self._database,
                    self._outdirs['models_dir'],
                    self._outdirs['results_dir'],
                    self._batch_size,
                    self._training_epochs,
                    self._k,
                    taxa,
                    self._merged_database_host['kmers'],
                    self._verbose
                )
            else:
                self.models[taxa] = KerasTFModel(
                    self._classifier_binary,
                    self._database,
                    self._outdirs['models_dir'],
                    self._outdirs['results_dir'],
                    self._batch_size,
                    self._training_epochs,
                    self._k,
                    taxa,
                    self._merged_database_host['kmers'],
                    self._verbose
                )
        self.models[taxa].preprocess(self._merged_training_datasets['train'])
        self.models[taxa].train(self._merged_training_datasets, self._merged_database_host, self._cv)

        self._save_model(self._model_file, taxa)

    def _multiclass_training(self, taxa):
        print('_multiclass_training')
        self._verify_classifier_multiclass()
        self._load_training_data()
        if self._classifier_multiclass in ['sgd','mnb']:
            self.models[taxa] = SklearnModel(
                self._classifier_multiclass,
                self._database,
                self._outdirs['models_dir'],
                self._outdirs['results_dir'],
                self._batch_size,
                self._training_epochs,
                self._k,
                taxa,
                self._database_data['kmers'],
                self._verbose
            )
        else:
            self.models[taxa] = KerasTFModel(
                self._classifier_multiclass,
                self._database,
                self._outdirs['models_dir'],
                self._outdirs['results_dir'],
                self._batch_size,
                self._training_epochs,
                self._k,
                taxa,
                self._database_data['kmers'],
                self._verbose
            )
        self.models[taxa].preprocess(self._training_datasets['train'])
        self.models[taxa].train(self._training_datasets, self._database_data, self._cv)
        self._save_model(self._model_file, taxa)

    # Private predicting secondary functions
    #########################################################################################################
# TODO: Revise these functions to parallelise with Ray + ease process
    # Classify sequences for first iteration
    def _classify_first(self, df, taxa, ids, df_file):
        print('_classify_first')
        try:
            pred_df = self._predict_sequences(df, taxa, ids)
            not_pred_df = pred_df[pred_df[taxa] == 'unknown']
            pred_df = pred_df[pred_df[taxa] != 'unknown']

            self.classified_data['classified_ids'] = list(pred_df['id'].values)
            self.classified_data['unknown_ids'] = list(not_pred_df['id'].values)

            self.classified_data['classification'] = pred_df

            if taxa == 'domain':
                if self._host == True:
                    pred_df_host = pred_df[pred_df['domain'] == 'host']
                    pred_df = pred_df[pred_df['domain'] != 'host']
                    classified_host, classified_host_file = self._extract_subset(df, df_file, list(pred_df_host['id'].values), taxa, 'bacteria')
                    self.classified_data[taxa]['host'] = {
                        'classification' : classified_host_file
                    }
                classified, classified_file = self._extract_subset(df, df_file, self.classified_data['classified_ids'], taxa, 'bacteria')
                self.classified_data[taxa]['bacteria'] = classified_file
                not_classified, not_classified_file = self._extract_subset(df, df_file, self.classified_data['unknown_ids'], taxa, 'unknown')
                self.classified_data[taxa]['unknown'] = not_classified_file
                return classified
            else:
                classified, classified_file = self._extract_subset(df, df_file, self.classified_data['classified_ids'], taxa, 'bacteria')
                self.classified_data[taxa]['classified'] = classified_file
                not_classified, not_classified_file = self._extract_subset(df, df_file, self.classified_data['unknown_ids'], taxa, 'unknown')
                self.classified_data[taxa]['unknown'] = not_classified_file
                return classified
        except:
            raise ValueError('No sequences to classify for {}.'.format(taxa))

    # Classify sequences according to passed taxa and model
    def _classify_subsequent(self, df, taxa, ids, df_file):
        print('_classify_subsequent')
        try:
            pred_df = self._predict_sequences(df, taxa, ids)
            not_pred_df = pred_df[pred_df[taxa] == 'unknown']
            pred_df = pred_df[pred_df[taxa] != 'unknown']

            self.classified_data['classification'] = self.classified_data['classification'].join(pred_df, how = 'outer', on = 'id')

            classified, classified_file = self._extract_subset(df, df_file, list(pred_df['id'].values), taxa, 'classified')
            self.classified_data[taxa]['classified'] = classified_file
            not_classified, not_classified_file = self._extract_subset(df, df_file, list(not_pred_df['id'].values), taxa, 'unknown')
            self.classified_data[taxa]['unknown'] = not_classified_file
            
            return classified
        except:
            raise ValueError('No sequences to classify for {}.'.format(taxa))

    # Make predictions
    def _predict_sequences(self, df, taxa, ids):
        print('_predict_sequences')
        try:
            predictions = self.models[taxa].predict(df, self._threshold)
            pred_df = pd.DataFrame({'id': ids, taxa: predictions.values})

            taxa_pos = self.classified_data['sequence'].index(taxa)
            lst_taxa = self.classified_data['sequence'][taxa_pos:]
            db_df = pd.DataFrame(
                self._database_data['classes'],
                columns=self._database_data['taxas']
            )[lst_taxa]
            pred_df = pred_df.merge(db_df, on=taxa, how='left')
            
            return pred_df
        except ValueError:
            raise ValueError('No sequences to classify for {}.'.format(taxa))

    # Extract subset of classified or not classified sequences
    def _extract_subset(self, df, df_file, ids, taxa, status):
        print('_extract_subset')
        clf_file = df_file + '_{}_{}'.format(taxa, status)
        rows_clf = []
        for row in df.iter_rows():
            if row['id'] in ids:
                rows_clf.append(row)
        df_clf = ray.data.from_items(rows_clf)
        if df_clf.count() > 0:
            df_clf.write_parquet(clf_file)
        return df_clf, clf_file

    # Helper functions
    #########################################################################################################

