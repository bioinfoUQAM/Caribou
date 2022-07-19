import numpy as np
import modin.pandas as pd

from abc import ABC, abstractmethod
from math import ceil

from models.build_neural_networks import *

import os
import ray
import json
import warnings

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDOneClassSVM, SGDClassifier

from joblib import parallel_backend, dump
from ray.util.joblib import register_ray

from keras.callbacks import EarlyStopping

from tensorflow import distribute, int64, TensorSpec
from tensorflow.config import list_physical_devices
from ray.train import Trainer, save_checkpoint, CheckpointStrategy
from ray.ml.predictors.integrations.tensorflow import TensorflowPredictor

__author__ = 'Nicolas de Montigny'

__all__ = ['ModelsUtils','SklearnModel','KerasTFModel','BatchInferModel']

# Ignore warnings to have a more comprehensible output on stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

register_ray()

class ModelsUtils(ABC):
    """
    Utilities for both types of framework

    ----------
    Attributes
    ----------

    k : int
        The length of K-mers extracted

    classifier : string
        The name of the classifier to be used

    outdir : string
        Path to a folder to output results

    batch_size : int
        Size of the batch used for online learning

    taxa : string
        The taxa for which the model is trained in classifying

    labels_list : list of int
        A list of the labels for multiclass learning / classifying

    ----------
    Methods
    ----------

    train : only train or cross-validate training of classifier
        X : ray.data.Dataset
            Dataset containing the K-mers profiles of sequences for learning
        y : ray.data.Dataset
            Dataset containing the classes of sequences for learning
        cv : boolean
            Should cross-validation be verified or not.
            Defaults to True.

    predict : abstract method to predict the classes of a dataset

    """
    def __init__(self, classifier, outdir_results, batch_size, k, taxa, verbose):
        # Parameters
        self.classifier = classifier
        self.outdir_results = outdir_results
        self.batch_size = batch_size
        self.k = k
        self.taxa = taxa
        self.verbose = verbose
        # Initialize empty
        self._label_encoder = None
        self.labels_list = []
        self._ids_list = []
        self._nb_kmers = 0
        # Files
        self._cv_csv = os.path.join(self.outdir_results,'{}_{}_K{}_cv_scores.csv'.format(self.classifier, self.taxa, self.k))

    def _preprocess(self, df):
        print('_preprocess')
        df = df.to_modin()
        self._ids_list = list(df.index)
        df = df.fillna(0)
        cols = df.columns
        self._nb_kmers = len(cols)
        if self.classifier != 'mnb':
            with parallel_backend('ray'):
                scaler = StandardScaler()
                df = pd.DataFrame(scaler.fit_transform(df), columns = cols)
        #with parallel_backend('ray'):
            #select = VarianceThreshold(threshold=0.9)
            #df = select.fit_transform(df)
            #df = pd.DataFrame(df, columns = select.get_feature_names_out())


        return ray.data.from_modin(df)

    @abstractmethod
    def _build(self):
        """
        """

    def train(self, X, y, cv = True):
        print('train')
        if cv:
            self._cross_validation(X, y)
        else:
            self._fit_model(X, y)

    @abstractmethod
    def _fit_model(self):
        """
        """

    def _cross_validation(self, X_train, y_train):
        print('_cross_validation')
        X_train = X_train.to_modin()
        y_train = y_train.to_modin()

        with parallel_backend('ray'):
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size = 0.8, random_state=42)

        X_train = ray.data.from_modin(X_train)
        y_train = ray.data.from_modin(y_train)
        X_test = ray.data.from_modin(X_test)
        y_test = ray.data.from_modin(y_test)

        self._fit_model(X_train, y_train)

        y_pred = self.predict(X_test)
        self._cv_score(y_test, y_pred)

    # Outputs scores for cross validation in a dictionnary
    def _cv_score(self, y_true, y_pred):
        print('_cv_score')

        support = []
        if self.classifier in ['onesvm','linearsvm', 'attention','lstm','deeplstm']:
            support = precision_recall_fscore_support(y_true.to_modin(), y_pred['classes'], pos_label = 'bacteria', average = 'binary')
        elif self.classifier in ['sgd','svm','mlr','mnb','lstm_attention','cnn','widecnn']:
            support = precision_recall_fscore_support(y_true.to_modin(), y_pred['classes'], average = 'macro')

        scores = pd.DataFrame({'Classifier':self.classifier,'Precision':support[0],'Recall':support[1],'F-score':support[2]}, index = [1]).T

        scores.to_csv(self._cv_csv, header = False)

    @abstractmethod
    def predict(self):
        """
        """

    def _label_encode(self, df):
        print('_label_encode')
        df = df.to_modin()
        with parallel_backend('ray'):
            self._label_encoder = LabelEncoder()
            df[self.taxa] = self._label_encoder.fit_transform(df[self.taxa])

        self.labels_list = np.unique(df[self.taxa])
        return ray.data.from_modin(df)

    def _label_decode(self, arr):
        print('_label_decode')
        decoded = np.empty(len(arr), dtype = object)
        decoded[arr == -1] = 'unknown'
        arr[decoded == 'unknown'] = 0
        with parallel_backend('ray'):
            arr = self._label_encoder.inverse_transform(arr)
        for pos in np.arange(len(decoded)):
            if decoded[pos] != 'unknown':
                decoded[pos] = arr[pos]

        return decoded

class SklearnModel(ModelsUtils):
    """
    Class used to build, train and predict models using Ray with Scikit-learn backend

    ----------
    Attributes
    ----------

    clf_file : string
        Path to a file containing the trained model for this object

    ----------
    Methods
    ----------

    predict : predict the classes of a dataset
        df : ray.data.Dataset
            Dataset containing K-mers profiles of sequences to be classified

    threshold : float
        Minimum percentage of probability to effectively classify.
        Sequences will be classified as 'unknown' if the probability is under this threshold.
        Defaults to 80%

    """
    def __init__(self, classifier, dataset, outdir_model, outdir_results, batch_size, k, taxa, verbose):
        super().__init__(classifier, outdir_results, batch_size, k, taxa, verbose)
        # Parameters
        if classifier in ['onesvm','linearsvm']:
            self.clf_file = '{}bacteria_binary_classifier_K{}_{}_{}_model.jb'.format(outdir_model, k, classifier, dataset)
        else:
            self.clf_file = '{}{}_multiclass_classifier_K{}_{}_{}_model.jb'.format(outdir_model, taxa, k, classifier, dataset)
        # Computes
        self._build()

    def _build(self):
        print('_build')
        if self.classifier == 'onesvm':
            if self.verbose:
                print('Training bacterial extractor with One Class SVM')
            self._clf = SGDOneClassSVM(nu = 0.05, tol = 1e-4)
        elif self.classifier == 'linearsvm':
            if self.verbose:
                print('Training bacterial / host classifier with Linear SVM')
            self._clf = SGDClassifier(early_stopping = False, n_jobs = -1)
        elif self.classifier == 'sgd':
            if self.verbose:
                print('Training multiclass SGD classifier with squared loss function (Ridge)')
            self._clf = SGDClassifier(loss = 'squared_error', n_jobs = -1, random_state = 42)
        elif self.classifier == 'svm':
            if self.verbose:
                print('Training multiclass SGD classifier with hinge loss (Linear SVM)')
            self._clf = SGDClassifier(loss = 'hinge', n_jobs = -1, random_state = 42)
        elif self.classifier == 'mlr':
            if self.verbose:
                print('Training multiclass Multinomial Logistic Regression classifier')
            self._clf = SGDClassifier(loss = 'log_loss', n_jobs = -1, random_state = 42)
        elif self.classifier == 'mnb':
            if self.verbose:
                print('Training multiclass Multinomial Naive Bayes classifier')
            self._clf = MultinomialNB()

    def _fit_model(self, X, y):
        print('_fit_model')
        X = self._preprocess(X)
        y = self._label_encode(y)

        with parallel_backend('ray'):
            if self.classifier == 'onesvm':
                for batch in X.iter_batches(batch_size = self.batch_size):
                        self._clf.partial_fit(batch)
            else:
                nb_batches = ceil(len(self._ids_list)/self.batch_size) - 1
                for iter, (batch_X, batch_y) in enumerate(zip(X.iter_batches(batch_size = self.batch_size), y.iter_batches(batch_size = self.batch_size))):
                    if iter != nb_batches:
                        self._clf.partial_fit(batch_X, batch_y, classes = self.labels_list)
                    else:
                        self._clf = CalibratedClassifierCV(base_estimator = self._clf, cv = 'prefit')
                        self._clf.fit(batch_X, batch_y)

        dump(self._clf, self.clf_file)

    def predict(self, df, threshold = 0.8):
        print('predict')
        y_pred = pd.DataFrame(columns = ['id','classes'])
        y_pred['id'] = df.to_modin()['id']
        df = self._preprocess(df)
        if self.classifier in ['onesvm','linearsvm']:
            y_pred['classes'] = self._predict_binary(df)
        elif self.classifier in ['sgd','svm','mlr','mnb']:
            y_pred['classes'] = self._predict_multi(df, threshold)

        return y_pred

    def _predict_binary(self, df):
        y_pred = np.empty(df.count(), dtype=np.int32)

        with parallel_backend('ray'):
            for i, row in enumerate(df.iter_batches(batch_size = 1)):
                y_pred[i] = self._clf.predict(row)

        if self.classifier == 'onesvm':
            return self._label_decode_onesvm(y_pred)
        else:
            return self._label_decode(y_pred)

    def _predict_multi(self, df, threshold):
        print('_predict_multi')
        y_pred = np.empty(df.count(), dtype=np.int32)
        with parallel_backend('ray'):
            for i, row in enumerate(df.iter_batches(batch_size = 1)):
                predicted = self._clf.predict_proba(row)
                print('predicted', predicted)
                print('np.argmax(predicted[0]) : ',np.argmax(predicted[0]))
                print('predicted[0,np.argmax(predicted[0])] : ',predicted[0,np.argmax(predicted[0])])
                if np.isnan(predicted[0,np.argmax(predicted[0])]):
                    y_pred[i] = -1
                elif predicted[0,np.argmax(predicted[0])] >= threshold:
                    y_pred[i] = np.argmax(predicted[0])
                else:
                    y_pred[i] = -1

        print('y_pred :')
        print(y_pred)

        return self._label_decode(y_pred)

    def _label_decode_onesvm(self, arr):
        decoded = np.empty(len(arr), dtype = object)
        decoded[arr == 1] = 'bacteria'
        decoded[arr == -1] = 'unknown'

        return decoded


class KerasTFModel(ModelsUtils):
    """
    Class used to build, train and predict models using Ray with Keras Tensorflow backend

    ----------
    Attributes
    ----------

    clf_file : string
        Path to a file containing the trained model for this object

    nb_classes : int
        Number of classes for learning

    ----------
    Methods
    ----------

    predict : predict the classes of a dataset
        df : ray.data.Dataset
            Dataset containing K-mers profiles of sequences to be classified

    threshold : float
        Minimum percentage of probability to effectively classify.
        Sequences will be classified as 'unknown' if the probability is under this threshold.
        Defaults to 80%

    """
    def __init__(self, classifier, dataset, outdir_model, outdir_results, batch_size, training_epochs, k, taxa, verbose):
        super().__init__(classifier, outdir_results, batch_size, k, taxa, verbose)
        # Parameters
        self.dataset = dataset
        self.outdir_model = outdir_model
        if classifier in ['attention','lstm','deeplstm']:
            self.clf_file = '{}bacteria_binary_classifier_K{}_{}_{}_model'.format(outdir_model, k, classifier, dataset)
        else:
            self.clf_file = '{}{}_multiclass_classifier_K{}_{}_{}_model'.format(outdir_model, taxa, k, classifier, dataset)
        # # Initialize empty
        self.nb_classes = None
        # Variables for training with Ray
        self._training_epochs = training_epochs
        self._strategy = distribute.MultiWorkerMirroredStrategy()
        if len(list_physical_devices('GPU')) > 0:
            self._trainer = Trainer(backend = 'tensorflow', num_workers = len(list_physical_devices('GPU')), use_gpu = True)
        else:
            self._trainer = Trainer(backend = 'tensorflow', num_workers = os.cpu_count())

    def _build(self, nb_kmers, nb_classes):
        print('_build')
        with self._strategy.scope():
            if self.classifier == 'attention':
                if self.verbose:
                    print('Training bacterial / host classifier based on Attention Weighted Neural Network')
                self._clf = build_attention(self.batch_size, self.k, nb_kmers)
            elif self.classifier == 'lstm':
                if self.verbose:
                    print('Training bacterial / host classifier based on Shallow LSTM Neural Network')
                self._clf = build_LSTM(self.k, self.batch_size)
            elif self.classifier == 'deeplstm':
                if self.verbose:
                    print('Training bacterial / host classifier based on Deep LSTM Neural Network')
                self._clf = build_deepLSTM(self.k, self.batch_size)
            elif self.classifier == 'lstm_attention':
                if self.verbose:
                    print('Training multiclass classifier based on Deep Neural Network hybrid between LSTM and Attention')
                self._clf = build_LSTM_attention(nb_kmers, nb_classes, self.batch_size)
            elif self.classifier == 'cnn':
                if self.verbose:
                    print('Training multiclass classifier based on CNN Neural Network')
                self._clf = build_CNN(self.k, self.batch_size, self.nb_classes)
            elif self.classifier == 'widecnn':
                if self.verbose:
                    print('Training multiclass classifier based on Wide CNN Network')
                self._clf = build_wideCNN(self.k, self.batch_size, self.nb_classes)

    def _fit_model(self, X, y):
        print('_fit_model')
        X = self._preprocess(X)
        y = self._label_encode(y)
        self.nb_classes = len(self.labels_list)

        checkpoint_strategy = CheckpointStrategy(num_to_keep = 1,
                                    checkpoint_score_attribute='val_accuracy',
                                    checkpoint_score_order='max')

        self._trainer.start()
        self._trainer.run(self._train_func, config = {'X':X,'y':y,'batch_size':self.batch_size,'epochs':self._training_epochs,'ids':self._ids_list,'nb_kmers':self._nb_kmers,'nb_classes':self.nb_classes}, checkpoint_strategy = checkpoint_strategy)
        self.checkpoint = self._trainer.best_checkpoint
        self._trainer.shutdown()

    def _train_func(self, config):
        print('_train_func')
        tf_config = json.loads(os.environ['TF_CONFIG'])
        num_workers = len(tf_config['cluster']['worker'])
        global_batch_size = config['batch_size'] * num_workers
        multi_worker_dataset = self._join_shuffle_data(config['X'], config['y'], global_batch_size, config['ids'], config['nb_kmers'])
        self._build(config['nb_kmers'], config['nb_classes'])
        early = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
        history = self._clf.fit(multi_worker_dataset, epochs = config['epochs'])
        print(history.history)
        save_checkpoint(model_weights = self._clf.get_weights())

    def _join_shuffle_data(self, X_train, y_train, batch_size, ids_list, nb_kmers):
        print('_join_shuffle_data')
        # Join
        X_train = X_train.to_modin()
        y_train = y_train.to_modin()
        X_train['id'] = ids_list
        y_train['id'] = ids_list
        df = X_train.merge(y_train, on = 'id', how = 'left')
        df = df.drop('id', 1)
        df = ray.data.from_modin(df)
        df = df.random_shuffle()
        df = df.to_tf(
        label_column = self.taxa,
        batch_size = batch_size,
        output_signature = (
            TensorSpec(shape=(None, nb_kmers), dtype=int64),
            TensorSpec(shape=(None,), dtype=int64),))

        return df

    def predict(self, df, threshold = 0.8):
        print('predict')
        y_pred = pd.DataFrame(columns = ['id','classes'])
        y_pred['id'] = df.to_modin()['id']
        df = self._preprocess(df)
        if self.classifier in ['attention','lstm','deeplstm']:
            y_pred['classes'] = self._predict_binary(df)
        elif self.classifier in ['lstm_attention','cnn','widecnn']:
            y_pred['classes'] = self._predict_multi(df, threshold)

        return y_pred

    def _predict_binary(self, df):
        predictor = TensorflowPredictor().from_checkpoint(checkpoint = self.checkpoint, model_definition = self._build)

        predicted = np.array(predictor.predict(df))

        y_pred = np.around(predicted.reshape(1, predicted.size)[0]).astype(np.int64)

        return self._label_decode(y_pred)

    def _predict_multi(self, df, threshold):
        y_pred = np.empty(df.count(), dtype=np.int32)
        predictor = TensorflowPredictor().from_checkpoint(checkpoint = self.checkpoint, model_definition = self._build)

        predicted = np.array(predictor.predict(df))

        for i in range(len(predicted)):
            if np.isnan(predicted[i,np.argmax(predicted[i])]):
                y_pred[i] = -1
            elif predict[i,np.argmax(predicted[i])] >= threshold:
                y_pred[i] = np.argmax(predicted[i])
            else:
                y_pred[i] = -1

        return self._label_decode(y_pred)

    def __reduce__(self):
        deserializer = self.__class__
        serialized_data = (self.classifier, self.dataset, self.outdir_model, self.outdir_results, self.batch_size, self._training_epochs, self.k, self.taxa, self.verbose)

        return deserializer, serialized_data
