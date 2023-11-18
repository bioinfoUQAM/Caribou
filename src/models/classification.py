import os
import ray
import cloudpickle

import numpy as np
import pandas as pd

from warnings import warn
from typing import Dict, List
from models.sklearn.models import SklearnModel
from models.kerasTF.models import KerasTFModel

# CV metrics
from sklearn.metrics import precision_recall_fscore_support

__author__ = 'Nicolas de Montigny'

__all__ = ['ClassificationMethods']

TRAINING_DATASET_NAME = 'train'
VALIDATION_DATASET_NAME = 'validation'
TEST_DATASET_NAME = 'test'
TENSOR_COLUMN_NAME = '__value__'

class ClassificationMethods():
    """
    Class for classifying sequences from metagenomes in a recursive manner

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
        db_data: Dict,
        outdirs: Dict,
        db_name: str,
        clf_binary: str = None,
        clf_multiclass: str = None,
        taxa: [str, List] = None,
        batch_size: int = 32,
        training_epochs: int = 100
    ):
        # Parameters
        self._taxas = taxa
        self._outdirs = outdirs
        self._database = db_name
        self._database_data = db_data
        self._classifier_binary = clf_binary
        self._classifier_multiclass = clf_multiclass
        self._batch_size = batch_size
        self._training_epochs = training_epochs
        # Init not fitted
        self.is_fitted = False

    # Public functions
    #########################################################################################################

    def fit(self, datasets):
        """
        Public function to call the fitting method after validation of parameters
        """
        self._valid_assign_taxas()
        self._valid_classifier()
        tax_map = self._verify_model_trained()
        
        self._fit(datasets, tax_map)
        
    def predict(self, dataset):
        """
        Public function to call the predicting method after validation of parameters
        """
        model_mapping = self._verify_load_model()
        predictions = self._predict(dataset, model_mapping)
        
        return predictions

    def fit_predict(self, datasets, predict_ds):
        """
        Public function for calling fit and predict after validation of parameters
        """
        self._valid_assign_taxas()
        self._valid_classifier()
        tax_map = self._verify_model_trained()

        self._fit(datasets, tax_map)

        model_mapping = self._verify_load_model()
        predictions = self._predict(predict_ds, model_mapping)
    
        return predictions

    def cross_validation(self, datasets):
        """
        Public function to call the cross-validation method after validation of parameters
        Executes cross-validation of a model by fitting it and predicting over a test dataset
        """        
        if isinstance(self._taxas, str):
            self._valid_assign_taxas()
            tax_map = self._verify_model_trained()

            test_ds = datasets.pop(TEST_DATASET_NAME)
            y_true, test_ds = self._get_true_classif(test_ds, self._taxas)
            
            self._fit(datasets, tax_map)

            model_mapping = self._verify_load_model()
            y_pred = self._cv_predict(test_ds, model_mapping)

            cv_scores = self._score_cv(y_true, y_pred, self._taxas[0])
            
            return cv_scores
        else:
            raise ValueError('Cross-validation can only be done on one taxa, please pass one taxa while initiating the ClassificationMethods object')


    # Private principal functions
    #########################################################################################################

    def _fit(self, datasets, tax_map):
        """
        Fit the given model to the training dataset
        """
        for taxa, file in tax_map.items():
            if taxa in ['domain','bacteria','host']:
                self._binary_training(datasets, taxa, file)
            else:
                self._multiclass_training(datasets, taxa, file)
        self.is_fitted = True

    def _predict(self, ds, model_map):
        """
        Predict the given data using the trained model in a recursive manner over taxas using a top-down approach
        Returns a mapping of the predictions made by the models for the targeted taxas
        """
        mapping = {}
        if self.is_fitted:
            try:
                for taxa, model in model_map.items():
                    predictions = model.predict(ds) # np.array
                    ds, predictions, ids = self._remove_unknown(ds, predictions)
                    file = self._save_dataset(ds, taxa)
                    mapping[taxa] = {
                        'classification' : predictions,
                        'ids' : ids,
                        'dataset' : file
                    }
                return mapping
            except ValueError:
                print('Stopping classification prematurelly because there are no more sequences to classify')
                return mapping
        else:
            raise ValueError('The model was not fitted yet! Please call either the `fit` or the `fit_predict` method before making predictions')

    def _cv_predict(self, ds, model_map):
        """
        Predict the given data using the trained model for cross-validation
        Returns a mapping of the predictions made by the models for the targeted taxas
        """
        mapping = {}
        for taxa, model in model_map.items():
            mapping[taxa] = model.predict(ds) # np.array
        return mapping

    # Private training secondary functions
    #########################################################################################################

    def _binary_training(self, datasets, taxa, file):
        print('_binary_training')
        if self._classifier_binary == 'onesvm':
            model = SklearnModel(
                self._classifier_binary,
                self._outdirs['models_dir'],
                self._batch_size,
                self._training_epochs,
                taxa,
                self._database_data['kmers']
            )
        elif self._classifier_binary == 'linearsvm':
            model = SklearnModel(
                self._classifier_binary,
                self._outdirs['models_dir'],
                self._batch_size,
                self._training_epochs,
                taxa,
                self._database_data['kmers']
            )
        else:
            model = KerasTFModel(
                self._classifier_binary,
                self._outdirs['models_dir'],
                self._batch_size,
                self._training_epochs,
                taxa,
                self._database_data['kmers']
            )
        model.preprocess(datasets[TRAINING_DATASET_NAME], os.path.join(self._outdirs['models_dir'], f'TruncatedSVD_components.npz'))
        model.fit(datasets)

        self._save_model(model, file)

    def _multiclass_training(self, datasets, taxa, file):
        print('_multiclass_training')
        if self._classifier_multiclass in ['sgd','mnb']:
            model = SklearnModel(
                self._classifier_multiclass,
                self._outdirs['models_dir'],
                self._batch_size,
                self._training_epochs,
                taxa,
                self._database_data['kmers']
            )
        else:
            model = KerasTFModel(
                self._classifier_multiclass,
                self._outdirs['models_dir'],
                self._batch_size,
                self._training_epochs,
                taxa,
                self._database_data['kmers']
            )
        model.preprocess(datasets[TRAINING_DATASET_NAME], os.path.join(self._outdirs['models_dir'], f'TruncatedSVD_components.npz'))
        model.fit(datasets)

        self._save_model(model, file)

    # Private predicting secondary functions
    #########################################################################################################

    def _remove_unknown(self, ds, predict):
        ids = []
        for row in ds.iter_rows():
            ids.append(row['id'])
        mapping = pd.DataFrame({
            'ids' : ids,
            'predictions' : predict
        })
        mapping = mapping[mapping['predictions'] != -1]
        ids = mapping['ids']
        predict = mapping['predictions']

        def remove_unknown(df):
            df = df[df['ids'].isin(ids)]
            return df
        
        ds = ds.map_batches(remove_unknown, batch_format = 'pandas')
        
        return ds, predict, ids

    # Private cross-validation secondary methods
    #########################################################################################################

    def _get_true_classif(self, ds, taxas):
        """
        Extract the true classification of the dataset used for cross-validation
        """
        classif = {taxa : [] for taxa in taxas}
        
        cols2drop = [col for col in ds.schema().names if col not in ['id', taxas[0]]]
        classif_ds = ds.drop_columns(cols2drop)

        cols2drop = [col for col in ds.schema().names if col not in ['id',TENSOR_COLUMN_NAME]]
        ds = ds.drop_columns(cols2drop)

        for row in classif_ds.iter_rows():
            for taxa in taxas:
                classif[taxa].append(row[taxa])

        return classif, ds

    def _score_cv(self, y_true, y_pred, taxa):
        """
        Compute the cross validation scores
        """
        if self._classifier_binary is not None:
            model = self._classifier_binary
        else :
            model = self._classifier_multiclass

        cv_csv = os.path.join(self._outdirs['results_dir'],f'{self._database}_{model}_{taxa}_cv_scores.csv')


        y_compare = pd.DataFrame({
            'y_true': y_true[taxa],
            'y_pred': y_pred[taxa]
        })
        y_compare['y_true'] = y_compare['y_true'].str.lower()
        y_compare['y_pred'] = y_compare['y_pred'].str.lower()
        y_compare.to_csv(os.path.join(self._outdirs['models_dir'], f'y_compare_{self._database}_{model}_{taxa}.csv'))

        support = precision_recall_fscore_support(
            y_compare['y_true'],
            y_compare['y_pred'],
            average = 'weighted'
        )

        scores = pd.DataFrame({
            taxa : [support[0],support[1],support[2]]
            },
            index = ['Precision','Recall','F-score']
        )
        
        scores.T.to_csv(cv_csv, index = True)

        return scores
    
    # Validation & verification methods
    #########################################################################################################

    def _valid_assign_taxas(self):
        """
        Validate taxas and assign to class variable
        Assign order for top-down strategy
        """
        print('_valid_assign_taxas')
        if self._taxas is None:
            self._taxas = self._database_data['taxas'].copy()            
        elif isinstance(self._taxas, list):
            self._taxas = self._taxas
        elif isinstance(self._taxas, str):
            self._taxas = [self._taxas]
        else:
            raise ValueError("Invalid taxa option, it must either be absent/None, be a list of taxas to extract or a string identifiying a taxa to extract")
        self._valid_taxas()
        self._taxas = [taxa for taxa in self._database_data['taxas'] if taxa in self._taxas]
        self._taxas.reverse()

    def _valid_taxas(self):
        """
        Validate that selected taxas are in database
        """
        print('_valid_taxas')
        for taxa in self._taxas:
            if taxa not in self._database_data['taxas']:
                raise ValueError("Taxa {} not found in database".format(taxa))

    def _valid_classifier(self):
        if self._classifier_binary is not None:
            if self._classifier_binary not in ['onesvm','linearsvm','attention','lstm','deeplstm']:
                raise ValueError("""
                                 Invalid classifier option for bacteria extraction!
                                 Models implemented at this moment are :
                                 Classic algorithm : One-class SVM (onesvm) and Linear SVM (linearsvm)
                                 Neural networks : Attention (attention), LSTM (lstm) and Deep LSTM (deeplstm)
                                 """)
        if self._classifier_multiclass is not None:
            if self._classifier_multiclass not in ['sgd','mnb','lstm_attention','cnn','widecnn']:
                raise ValueError("""
                                 Invalid classifier option for bacteria classification!
                                 Models implemented at this moment are :
                                 Classic algorithm : Stochastic Gradient Descent (sgd) and Multinomial Naïve Bayes (mnb)
                                 Neural networks : Deep hybrid between LSTM and Attention (lstm_attention), CNN (cnn) and Wide CNN (widecnn)
                                 """)

    def _verify_model_trained(self):
        """
        Verify if the model is already trained for all desired taxas
        Taxas for which a model is already trained will be removed from the list
        Returns a mapping of the file per taxa to train
        """
        mapping = {}
        for taxa in self._taxas:
            if taxa in ['domain','bacteria','host']:
                clf = self._classifier_binary
            else:
                clf = self._classifier_multiclass
            file = os.path.join(self._outdirs['models_dir'], f'{clf}_{taxa}.pkl')
            if not os.path.isfile(file):
                mapping[taxa] = file
        
        return mapping
    
    def _verify_load_model(self):
        """
        Verify if the model is already trained for all desired taxas
        Taxas for which no model was not trained will raise a ValueError
        Returns a mapping of the model per taxa for predicting
        """
        mapping = {}
        for taxa in self._taxas:
            if taxa in ['domain','bacteria','host']:
                clf = self._classifier_binary
            else:
                clf = self._classifier_multiclass
            file = os.path.join(self._outdirs['models_dir'], f'{clf}_{taxa}.pkl')
            if not os.path.isfile(file):
                raise ValueError(f'No model found for {taxa}')
            else:
                mapping[taxa] = self._load_model(file, taxa)
        return mapping
    
    def _load_model(self, file, taxa):
        """
        Load a model from the specified file
        """
        print('_load_model')
        with open(file, 'rb') as handle:
            return cloudpickle.load(handle)
        
    def _save_model(self, model, file):
        """
        Save a model to a specified file
        """
        print('_save_model')
        with open(file, 'wb') as handle:
            cloudpickle.dump(model, handle)
    
    def _save_dataset(self, ds, taxa):
        """
        Save a dataset to disk and return the filename
        """
        if taxa in ['domain','bacteria','host']:
            model = self._classifier_binary
        else:
            model = self._classifier_multiclass
        file = os.path.join(self._outdirs['results'], f'data_classified_{model}_{taxa}.parquet')
        ds.write_parquet(file)
        return file