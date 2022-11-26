import os
import ray
import cloudpickle

import pandas as pd

from utils import load_Xy_data
from models.ray_sklearn import SklearnModel
from models.ray_keras_tf import KerasTFModel

__author__ = 'Nicolas de Montigny'

__all__ = ['ClassificationMethods']

class ClassificationMethods():
    """
    Utilities class for classifying sequences from metagenomes using ray

    ----------
    Attributes
    ----------

    ----------
    Methods
    ----------

    """
    def __init__(
        self,
        database_k_mers,
        k,
        outdirs,
        database,
        classifier_binary = 'deeplstm',
        classifier_multiclass = 'lstm_attention',
        taxa = None,
        threshold = 0.8,
        batch_size = 32,
        training_epochs = 100,
        verbose = True,
        cv = False
    ):
        # Parameters
        self.k = k
        self.cv = cv
        self.outdirs = outdirs
        self.database = database
        self.verbose = verbose
        self.threshold = threshold
        self.classifier_binary = classifier_binary
        self.classifier_multiclass = classifier_multiclass
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        # Initialize with values
        self.classified_data = {'order': []}
        # Empty initializations
        self.taxas = []
        self.models = {}
        self.host = False
        self.X_train = None
        self.y_train = None
        self.classified_ids = []
        self.host_data = None
        self.database_data = None
        self.not_classified_ids = []
        self.merged_database_host = None
        self.previous_taxa_unclassified = None
        if isinstance(database_k_mers, tuple):
            self.host = True
            self.database_data = database_k_mers[0]
            self.host_data = database_k_mers[1]
        else:
            self.database_data = database_k_mers
        # Automatic executions
        self._verify_assign_taxas(taxa)
        
    # Main functions
    #########################################################################################################

    # Execute training of model(s)
    def execute_training(self):
        for taxa in self.taxas:
            if taxa in ['domain','bacteria','host']:
                clf = self.classifier_binary
            else:
                clf = self.classifier_multiclass
            self._data_file = os.path.join(
                self.outdirs['data_dir'],
                'Xy_{}_database_K{}_{}_{}_data.npz'.format(
                    taxa,
                    self.k,
                    clf,
                    self.database
                )
            )
            self._model_file = os.path.join(
                self.outdirs['models_dir'],
                '{}_{}.pkl'.format(
                    clf,
                    taxa
                )
            )
            train = self._verify_load_data_model(self._data_file, self._model_file, taxa)
            if train:
                self._train_model(taxa)

    # Train model according to passed taxa
    def _train_model(self, taxa):
        if taxa in ['domain','bacteria','host']:
            self._binary_training(taxa)
        else:
            self._multiclass_training(taxa)

    def _binary_training(self, taxa):
        self._verify_classifier_binary()
        if self.classifier_binary == 'onesvm':
            self.models[taxa] = SklearnModel(
                self.classifier_binary,
                self.database,
                self.outdirs['models_dir'],
                self.outdirs['results_dir'],
                self.batch_size,
                self.k,
                taxa,
                self.database_data['kmers'],
                self.verbose
            )
            self.X_train = ray.data.read_parquet(self.database_data['profile'])
            self.y_train = pd.DataFrame(
                {taxa: pd.DataFrame(self.database_data['classes'], columns=self.database_data['taxas']).loc[:, taxa].astype('string').str.lower(),
                 'id': self.database_data['ids']}
            )
        else:
            self._merge_database_host(self.database_data, self.host_data)
            if self.classifier_binary == 'linearSVM':
                self.models[taxa] = SklearnModel(
                    self.classifier_binary,
                    self.database,
                    self.outdirs['models_dir'],
                    self.outdirs['results_dir'],
                    self.batch_size,
                    self.k,
                    taxa,
                    self.self.merged_database_host['kmers'],
                    self.verbose
                )
            else:
                self.models[taxa] = KerasTFModel(
                    self.classifier_binary,
                    self.database,
                    self.outdirs['models_dir'],
                    self.outdirs['results_dir'],
                    self.batch_size,
                    self.training_epochs,
                    self.k,
                    taxa,
                    self.self.merged_database_host['kmers'],
                    self.verbose
                )
            self.X_train = ray.data.read_parquet(self.database_data['profile'])
            self.y_train = pd.DataFrame(
                {taxa: pd.DataFrame(self.database_data['classes'], columns=self.database_data['taxas']).loc[:, taxa].astype('string').str.lower(),
                 'id': self.database_data['ids']}
            )
        self.models[taxa].train(self.X_train, self.y_train, self.database_data, self.cv)
        self._save_model(self._model_file, taxa)

    def _multiclass_training(self, taxa):
        self._verify_classifier_multiclass()
        if self.classifier_multiclass in ['sgd','mnb']:
            self.models[taxa] = SklearnModel(
                self.classifier_multiclass,
                self.database,
                self.outdirs['models_dir'],
                self.outdirs['results_dir'],
                self.batch_size,
                self.k,
                taxa,
                self.database_k_mers['kmers'],
                self.verbose
            )
        else:
            self.models[taxa] = KerasTFModel(
                self.classifier_multiclass,
                self.database,
                self.outdirs['models_dir'],
                self.outdirs['results_dir'],
                self.batch_size,
                self.training_epochs,
                self.k,
                taxa,
                self.database_k_mers['kmers'],
                self.verbose
            )
        self.X_train = ray.data.read_parquet(self.database_data['profile'])
        self.y_train = pd.DataFrame(
            {taxa: pd.DataFrame(self.database_data['classes'], columns=self.database_data['taxas']).loc[:, taxa].astype('string').str.lower(),
             'id': self.database_data['ids']}
        )
        self.models[taxa].train(self.X_train, self.y_train, self.database_data, self.cv)
        self._save_model(self._model_file, taxa)
        
    # Execute classification using trained model(s)
    def execute_classification(self, data2classify):
        df_file = data2classify['profile']
        df = ray.data.read_parquet(df_file)
        ids = data2classify['ids']
        for i, taxa in enumerate(self.classified_data['order']):
            if i == 0:
                df = self._classify_first(df, taxa, ids, df_file)
            else:
                df = self._classify_subsequent(df, taxa, ids, df_file)

    # Classify sequences for first iteration
    def _classify_first(self, df, taxa, ids, df_file):
        pred_df = self._predict_sequences(df, taxa, ids)
        
        self.classified_ids = list(pred_df['id'].values)
        self.not_classified_ids = list(np.setdiff1d(ids, self.classified_ids, assume_unique=True))

        if self.host == True:
            pred_df = pred_df[pred_df['domain'] == 'host']
            pred_df_host = pred_df[pred_df['domain'] != 'host']
            self.classified_data['host'] = pred_df_host

        self.classified_data[taxa] = pred_df
        not_classified = self._extract_subset_not_classified(df, df_file, taxa)
        
        return not_classified

    # Classify sequences according to passed taxa and model
    def _classify_subsequent(self, df, taxa, ids, df_file):
        pred_df = self._predict_sequences(df, taxa, ids)

        self.classified_ids = self.classified_ids.extend(list(pred_df['id'].values))
        self.not_classified_ids = [id for id in ids if id not in self.classified_ids]

        self.classified_data[taxa] = pred_df
        not_classified = self._extract_subset_not_classified(df, df_file, taxa)

        return not_classified

    # Make predictions
    def _predict_sequences(self, df, taxa, ids):
        predictions = self.models[taxa].predict(df, self.threshold)
        pred_df = pd.DataFrame({'id': ids, taxa: predictions.values})

        taxa_pos = self.classified_data['order'].index(taxa)
        lst_taxa = self.classified_data['order'][taxa_pos:]
        db_df = pd.DataFrame(
            self.database_data['classes'],
            columns=self.database_data['taxas']
        )[[lst_taxa]]
        pred_df = pred_df.merge(db_df, on=taxa, how='left')
        
        return pred_df

    # Extract subset of sequences not classified
    def _extract_subset_not_classified(self, df, df_file, taxa):
        not_classified_file = df_file + '_{}'.format(taxa)
        rows_not_classified = []
        df = ray.data.read_parquet(df)
        for row in df.iter_rows():
            if row['id'] in self.not_classified_ids:
                rows_not_classified.append(row)

        not_classified = ray.data.from_items(rows_not_classified)
        not_classified.write_parquet(not_classified_file)
        return not_classified

    # Utils functions
    #########################################################################################################
    
     # Merge database and host reference data for bacteria extraction training
    def _merge_database_host(self, database_data, host_data):
        self.merged_database_host = {}

        self.merged_database_host['profile'] = "{}_host_merged".format(os.path.splitext(database_data["profile"])[0]) # Kmers profile

        df_classes = pd.DataFrame(database_data["classes"], columns=database_data["taxas"])
        if len(np.unique(df_classes['domain'])) != 1:
            df_classes[df_classes['domain'] != 'bacteria'] = 'bacteria'
        df_classes = df_classes.append(pd.DataFrame(host_data["classes"], columns=host_data["taxas"]), ignore_index=True)
        self.merged_database_host['classes'] = np.array(df_classes)  # Class labels
        self.merged_database_host['kmers'] = database_data["kmers"]  # Features
        self.merged_database_host['taxas'] = database_data["taxas"]  # Known taxas for classification
        self.merged_database_host['fasta'] = (database_data['fasta'], host_data['fasta'])  # Fasta file needed for reads simulation

        df_db = ray.data.read_parquet(database_data["profile"])
        df_host = ray.data.read_parquet(host_data["profile"])
        df_merged = df_db.union(df_host)
        df_merged.write_parquet(self.merged_database_host['profile'])

    # Verify taxas and assign to class variable
    def _verify_assign_taxas(self, taxa):
        if taxa is None:
            self.taxas = self.database_data['taxas'].copy()
        elif isinstance(taxa, list):
            self.taxas = taxa
        elif isinstance(taxa, str):
            self.taxas = [taxa]
        else:
            raise ValueError("Invalid taxa option, it must either be absent/None, be a list of taxas to extract or a string identifiying a taxa to extract")
        self._verify_taxas()

    # Verify if selected taxas are in database
    def _verify_taxas(self):
        for taxa in self.taxas:
            if taxa not in self.database_data['taxas']:
                raise ValueError("Taxa {} not found in database".format(taxa))

    # Caller function for verifying if the data and model already exist
    def _verify_load_data_model(self, data_file, model_file, taxa):
        self._verify_files(data_file, taxa)
        return self._verify_load_model(model_file, taxa)
        
    # Load extracted data if already exists
    def _verify_files(self, file, taxa):
        self.classified_data['order'].append(taxa)
        if os.path.isfile(file):
            self.classified_data[taxa] = load_Xy_data(file)
        else:
            self.classified_data[taxa] = {}

    # Load model if already exists
    def _verify_load_model(self, model_file, taxa):
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                self.models[taxa] = cloudpickle.load(f)
            return False
        else:
            return True

    def _save_model(self, model_file, taxa):
        with open(model_file, 'wb') as f:
            cloudpickle.dump(self.models[taxa], f)

    def _verify_classifier_binary(self):
        if self.classifier_binary == 'onesvm' and self.host == True:
            raise ValueError('Classifier One-Class SVM cannot be used with host data!\nEither remove host data from config file or choose another bacteria extraction method')
        elif self.classifier_binary == 'onesvm' and self.host == False:
            pass
        elif self.classifier_binary in ['linearsvm','attention','lstm','deeplstm'] and self.host == True:
            pass
        elif self.classifier_binary in ['linearsvm','attention','lstm','deeplstm'] and self.host == False:
            raise ValueError('Classifier {} cannot be used without host data!\nEither add host data to config file or choose the One-Class SVM classifier'.format(self.classifier_binary))
        else:
            raise ValueError('Invalid classifier option for bacteria extraction!\n\tModels implemented at this moment are :\n\tBacteria isolator :  One Class SVM (onesvm)\n\tClassic algorithm : Linear SVM (linearsvm)\n\tNeural networks : Attention (attention), Shallow LSTM (lstm) and Deep LSTM (deeplstm)')

    def _verify_classifier_multiclass(self):
        if self.classifier_multiclass in ['sgd','mnb','lstm_attention','cnn','widecnn']:
            pass
        else:
            raise ValueError('Invalid classifier option for bacteria classification!\n\tModels implemented at this moment are :\n\tClassic algorithm : Stochastic Gradient Descent (sgd) and Multinomial Naïve Bayes (mnb)\n\tNeural networks : Deep hybrid between LSTM and Attention (lstm_attention), CNN (cnn) and Wide CNN (widecnn)')
