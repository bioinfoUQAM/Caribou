import os
import ray
import cloudpickle

import numpy as np
import pandas as pd

from utils import zip_X_y
from glob import glob
from shutil import rmtree
from utils import load_Xy_data
from models.sklearn.ray_sklearn import SklearnModel
from models.kerasTF.ray_keras_tf import KerasTFModel

# Simulation class
from models.reads_simulation import readsSimulation

__author__ = 'Nicolas de Montigny'

__all__ = ['ClassificationMethods']

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

    execute_training : launch the training of the models for the chosen taxonomic levels
        no parameters to pass

    execute_classification : 
        data2classify : a dictionnary containing the data to classify produced by the function Caribou.src.data.build_data.build_X_data

    """
    def __init__(
        self,
        database_k_mers,
        k,
        outdirs,
        database,
        classifier_binary = 'deeplstm',
        classifier_multiclass = 'widecnn',
        taxa = None,
        threshold = 0.8,
        batch_size = 32,
        training_epochs = 100,
        verbose = True,
        cv = False
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
        
    # Main functions
    #########################################################################################################

    # Wrapper function for training and predicting over each known taxa
    def execute_training_prediction(self, data2classify):
        print('execute_training_prediction')
        file2classify = data2classify['profile']
        
        df2classify = ray.data.read_parquet(file2classify)
        ids2classify = data2classify['ids']
        for i, taxa in enumerate(self._taxas_order):
            if taxa in self._taxas:
                # Training
                if taxa in ['domain','bacteria','host']:
                    clf = self._classifier_binary
                else:
                    clf = self._classifier_multiclass
                self._data_file = os.path.join(self._outdirs['data_dir'], f'Xy_{taxa}_database_K{self._k}_{clf}_{self._database}_data.npz')
                self._model_file = os.path.join(self._outdirs['models_dir'], f'{clf}_{taxa}.pkl')
                train = self._verify_load_data_model(self._data_file, self._model_file, taxa)
                if train:
                    self._train_model(taxa)
                # Predicting
                try:
                    if i == 0:
                        df2classify = self._classify_first(df2classify, taxa, ids2classify, file2classify)
                    else:
                        df2classify = self._classify_subsequent(df2classify, taxa, ids2classify, file2classify)
                except ValueError:
                    print('Stopping classification prematurelly because there are no more sequences to classify')
                    return taxa
        return None


    # Execute training of model(s)
    def execute_training(self):
        print('execute_training')
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
                    self._train_model(taxa)

    # Train model according to passed taxa
    def _train_model(self, taxa):
        print('_train_model')
        if taxa in ['domain','bacteria','host']:
            self._binary_training(taxa)
        else:
            self._multiclass_training(taxa)

    def _binary_training(self, taxa):
        print('_binary_training')
        self._verify_classifier_binary()
        self._load_training_data_merged(taxa)
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
        
    # Execute classification using trained model(s) over a given taxa
    def execute_classification(self, data2classify):
        print('execute_classification')
        df_file = data2classify['profile']
        df = ray.data.read_parquet(df_file)
        ids = data2classify['ids']
        if len(self.classified_data['sequence']) == 0:
            raise ValueError('Please train a model before executing classification')
        for i, taxa in enumerate(self.classified_data['sequence']):
            try:
                if i == 0:
                    df = self._classify_first(df, taxa, ids, df_file)
                else:
                    df = self._classify_subsequent(df, taxa, ids, df_file)
            except ValueError:
                print('Stopping classification prematurelly because there are no more sequences to classify')
                return taxa
        return None

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

    # Utils functions
    #########################################################################################################
    
    # Verify taxas and assign to class variable
    def _verify_assign_taxas(self, taxa):
        print('_verify_assign_taxas')
        if taxa is None:
            self._taxas = self._database_data['taxas'].copy()            
        elif isinstance(taxa, list):
            self._taxas = taxa
        elif isinstance(taxa, str):
            self._taxas = [taxa]
        else:
            raise ValueError("Invalid taxa option, it must either be absent/None, be a list of taxas to extract or a string identifiying a taxa to extract")
        self._verify_taxas()

    # Verify if selected taxas are in database
    def _verify_taxas(self):
        print('_verify_taxas')
        for taxa in self._taxas:
            if taxa not in self._database_data['taxas']:
                raise ValueError("Taxa {} not found in database".format(taxa))

    # Caller function for verifying if the data and model already exist
    def _verify_load_data_model(self, data_file, model_file, taxa):
        print('_verify_load_data_model')
        self._verify_files(data_file, taxa)
        return self._verify_load_model(model_file, taxa)
        
    # Load extracted data if already exists
    def _verify_files(self, file, taxa):
        print('_verify_files')
        self.classified_data['sequence'].append(taxa)
        if os.path.isfile(file):
            self.classified_data[taxa] = load_Xy_data(file)
        else:
            self.classified_data[taxa] = {}

    # Load model if already exists
    def _verify_load_model(self, model_file, taxa):
        print('_verify_load_model')
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                self.models[taxa] = cloudpickle.load(f)
            return False
        else:
            return True

    def _save_model(self, model_file, taxa):
        print('_save_model')
        with open(model_file, 'wb') as f:
            cloudpickle.dump(self.models[taxa], f)

    def _verify_classifier_binary(self):
        print('_verify_classifier_binary')
        if self._classifier_binary == 'onesvm':
            if self._cv == True and self._host == True:
                pass
            elif self._cv == True and self._host == False:
                raise ValueError('Classifier One-Class SVM cannot be cross-validated with bacteria data only!\nEither add host data from parameters or choose to predict directly using this method')
            elif self._cv == False and self._host == True:
                raise ValueError('Classifier One-Class SVM cannot classify with host data!\nEither remove host data from parameters or choose another bacteria extraction method')
            elif self._cv == False and self._host == False:
                pass
        elif self._classifier_binary == 'onesvm' and self._host == False:
            pass
        elif self._classifier_binary in ['linearsvm','attention','lstm','deeplstm'] and self._host == True:
            pass
        elif self._classifier_binary in ['linearsvm','attention','lstm','deeplstm'] and self._host == False:
            raise ValueError('Classifier {} cannot classify without host data!\nEither add host data to config file or choose the One-Class SVM classifier'.format(self._classifier_binary))
        else:
            raise ValueError('Invalid classifier option for bacteria extraction!\n\tModels implemented at this moment are :\n\tBacteria isolator :  One Class SVM (onesvm)\n\tClassic algorithm : Linear SVM (linearsvm)\n\tNeural networks : Attention (attention), Shallow LSTM (lstm) and Deep LSTM (deeplstm)')

    def _verify_classifier_multiclass(self):
        print('_verify_classifier_multiclass')
        if self._classifier_multiclass in ['sgd','mnb','lstm_attention','cnn','widecnn']:
            pass
        else:
            raise ValueError('Invalid classifier option for bacteria classification!\n\tModels implemented at this moment are :\n\tClassic algorithm : Stochastic Gradient Descent (sgd) and Multinomial Naïve Bayes (mnb)\n\tNeural networks : Deep hybrid between LSTM and Attention (lstm_attention), CNN (cnn) and Wide CNN (widecnn)')

    # Merge database and host reference data for bacteria extraction training
    def _merge_database_host(self, database_data, host_data):
        print('_merge_database_host')
        self._merged_database_host = {}
        self._merged_database_host['profile'] = f"{database_data['profile']}_host_merged" # Kmers profile

        if os.path.exists(self._merged_database_host['profile']):
            df_merged = ray.data.read_parquet(self._merged_database_host['profile'])
        else:
            df_db = ray.data.read_parquet(database_data['profile'])
            df_host = ray.data.read_parquet(host_data['profile'])

            cols2drop = []
            for col in df_db.schema().names:
                if col not in ['id','domain','__value__']:
                    cols2drop.append(col)
            df_db = df_db.drop_columns(cols2drop)
            cols2drop = []
            for col in df_host.schema().names:
                if col not in ['id','domain','__value__']:
                    cols2drop.append(col)
            df_host = df_host.drop_columns(cols2drop)

            df_merged = df_db.union(df_host)
            df_merged.write_parquet(self._merged_database_host['profile'])

        self._merged_database_host['ids'] = np.concatenate((database_data["ids"], host_data["ids"]))  # IDs
        self._merged_database_host['kmers'] = database_data["kmers"]  # Features
        self._merged_database_host['taxas'] = ['domain']  # Known taxas for classification
        self._merged_database_host['fasta'] = (database_data['fasta'], host_data['fasta'])  # Fasta file needed for reads simulation

        return df_merged

    # Load, merge db + host & simulate validation / test datasets
    def _load_training_data_merged(self, taxa):
        print('_load_training_data_merged')
        def convert_archaea_bacteria(df):
            df.loc[df['domain'] == 'Archaea', 'domain'] = 'Bacteria'
            return df
        if self._classifier_binary == 'onesvm' and taxa == 'domain':
            df_train = ray.data.read_parquet(self._database_data['profile'])
            df_val_test = self._merge_database_host(self._database_data, self._host_data)
            df_val_test = df_val_test.map_batches(convert_archaea_bacteria, batch_format = 'pandas')
            df_val = self.split_sim_cv_ds(df_val_test,self._merged_database_host, 'merged_validation')
            self._merged_training_datasets = {'train': df_train, 'validation': df_val}
            if self._cv:
                df_test = self.split_sim_cv_ds(df_val_test,self._merged_database_host, 'merged_test')
                self._merged_training_datasets['test'] = df_test
        else:
            df_train = self._merge_database_host(self._database_data, self._host_data)
            df_train = df_train.map_batches(convert_archaea_bacteria, batch_format = 'pandas')
            df_val = self.split_sim_cv_ds(df_train,self._merged_database_host, 'merged_validation')
            self._merged_training_datasets = {'train': df_train, 'validation': df_val}
            if self._cv:
                df_test = self.split_sim_cv_ds(df_train,self._merged_database_host, 'merged_test')
                self._merged_training_datasets['test'] = df_test

    # Load db & simulate validation / test datasets
    def _load_training_data(self):
        print('_load_training_data')
        def convert_archaea_bacteria(df):
            df.loc[df['domain'] == 'Archaea', 'domain'] = 'Bacteria'
            return df
        df_train = ray.data.read_parquet(self._database_data['profile'])
        df_train = df_train.map_batches(convert_archaea_bacteria, batch_format = 'pandas')
        df_val = self.split_sim_cv_ds(df_train,self._database_data, 'validation')
        self._training_datasets = {'train': df_train, 'validation': df_val}
        if self._cv:
            df_test = self.split_sim_cv_ds(df_train,self._database_data, 'test')
            self._training_datasets['test'] = df_test

    def _sim_4_cv(self, df, kmers_ds, name):
        print('_sim_4_cv')
        cols = ['id']
        cols.extend(kmers_ds['taxas'])
        cls = pd.DataFrame(columns = cols)
        for batch in df.iter_batches(batch_format = 'pandas'):
            cls = pd.concat([cls, batch[cols]], axis = 0, ignore_index = True)
        
        sim_outdir = os.path.dirname(kmers_ds['profile'])
        cv_sim = readsSimulation(kmers_ds['fasta'], cls, list(cls['id']), 'miseq', sim_outdir, name)
        sim_data = cv_sim.simulation(self._k, kmers_ds['kmers'])
        df = ray.data.read_parquet(sim_data['profile'])
        return df
    
    def split_sim_cv_ds(self, ds, data, name):
        ds_path = os.path.join(
            os.path.dirname(data['profile']),
            f'Xy_genome_simulation_{name}_data_K{len(data["kmers"][0])}'
            )
        if os.path.exists(ds_path):
            cv_ds = ray.data.read_parquet(ds_path)
        else:
            cv_ds = ds.random_sample(0.1)
            if cv_ds.count() == 0:
                nb_smpl = round(ds.count() * 0.1)
                cv_ds = ds.random_shuffle().limit(nb_smpl)
            cv_ds = self._sim_4_cv(ds, data, name)
        return cv_ds
    