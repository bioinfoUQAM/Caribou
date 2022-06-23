import numpy
import modin.pandas as pd

from abc import ABC, abstractmethod

import os
import ray
import json
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDOneClassSVM, SGDClassifier

from joblib import parallel_backend
from ray.util.joblib import register_ray

from keras.callbacks import EarlyStopping

from ray.ml.predictors.tensorflow import TensorflowPredictor
from ray.train import Trainer, save_checkpoint, CheckpointStrategy

__author__ = 'Nicolas de Montigny'

__all__ = ['ModelsUtils','SklearnModel','KerasTFModel']

# Ignore warnings to have a more comprehensible output on stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

register_ray()

class ModelsUtils(ABC):
    '''
    Utilities for both types of framework
    '''
    def __init__(classifier, outdir, batch_size, k, verbose):
        # Parameters
        self.classifier = classifier
        self.outdir = outdir
        self.batch_size = batch_size
        self.k = k
        self.verbose = verbose
        # Initialize empty
        self.label_encoder = None
        self.labels_list = []

    # Data scaling
    def _scaleX(df):
        kmers = list(df.limit(1).to_pandas().columns)
        kmers.remove('id')
        kmers.remove('classes')
        df = df.to_modin()
        with parallel_backend('ray'):
            scaler = StandardScaler()
            df[kmers] = scaler.fit_transform(df[kmers])

        return ray.data.from_modin(df)

    @abstractmethod
    def _build(self):
        """
        """

    def train(self, X, y, cv = True):
        if cv:
            self._cross_validation(X, y)
        else:
            self._fit_model(X, y)

    @abstractmethod
    def _fit_model(self):
        """
        """

    def _cross_validation(self, X_train, y_train):
        with parallel_backend('ray'):
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size = 0.8, random_state=42)

            self._fit_model(X_train, y_train)

            y_pred = self.predict(X_test)
            self._cv_score(y_test, y_pred)

    # Outputs scores for cross validation in a dictionnary
    def _cv_score(y_true, y_pred, classifier, k, outdir):

        if classifier in ['onesvm','linearsvm', 'attention','lstm','deeplstm']:
            average = 'binary'
        elif classifier in ['sgd','svm','mlr','mnb','lstm_attention','cnn','widecnn']:
            average = 'macro'

            support = precision_recall_fscore_support(y_true, y_pred , average = average)


            scores = pd.DataFrame({'Classifier':self.classifier,'Precision':support[0],'Recall':support[1],'F-score':support[2]})

            scores.to_csv(os.join(self.outdir,'{}_K{}_cv_scores.csv'.format(self.classifier, self.k)))

    @abstractmethod
    def predict(self):
        """
        """

    def _label_encode(self, df):
        df = df.to_modin()
        with parallel_backend('ray'):
            self.label_encoder = LabelEncoder()
            df['classes'] = self.label_encoder.fit_transform(df['classes'])

        self.labels_list = np.unique(df['classes'])
        return ray.data.from_modin(df)

    def _label_decode(self, df):
        df = df.to_modin()
        with parallel_backend('ray'):
            df['classes'] = self.label_encoder.inverse_transform(df['classes'])

        return ray.data.from_modin(df)

class SklearnModel(ModelsUtils):
    '''
    Class to be used to build, train and predict models using Ray with Scikit-learn backend
    '''
    def __init__(classifier, dataset, outdir_model, outdir_results, batch_size, k, verbose):
        super().__init__(classifier, dataset, outdir_results, batch_size, k, verbose)
        # Parameters
        self.clf_file = '{}bacteria_binary_classifier_K{}_{}_{}_model.jb'.format(outdir_model, k, classifier, dataset)
        # Computes
        self._build()


    def _build(self):
        if self.classifier == 'onesvm':
            if self.verbose:
                print('Training bacterial extractor with One Class SVM')
            self.clf = SGDOneClassSVM(nu = 0.05, tol = 1e-4)
        elif self.classifier == 'linearsvm':
            if self.verbose:
                print('Training bacterial / host classifier with Linear SVM')
            self.clf = SGDClassifier(early_stopping = False, n_jobs = -1)
        elif self.classifier == 'sgd':
            if self.verbose:
                print('Training multiclass classifier with SGD and squared loss function')
            self.clf = SGDClassifier(loss = 'squared_error', n_jobs = -1, random_state = 42)
        elif self.classifier == 'svm':
            if self.verbose:
                print('Training multiclass classifier with Linear SVM and SGD hinge loss')
            self.clf = SGDClassifier(loss = 'hinge', n_jobs = -1, random_state = 42)
        elif self.classifier == 'mlr':
            if self.verbose:
                print('Training multiclass classifier with Multinomial Logistic Regression')
            self.clf = SGDClassifier(loss = 'log', n_jobs = -1, random_state = 42)
        elif self.classifier == 'mnb':
            if self.verbose:
                print('Training multiclass classifier with Multinomial Naive Bayes')
            self.clf = MultinomialNB()

    def _fit_model(self, X, y):
        X = self.scaleX(X)
        y = _label_encode(y)
        with parallel_backend('ray'):
            if self.classifier == 'onesvm':
                for batch in X.iter_batches(batch_size = self.batch_size):
                    self.clf.partial_fit(batch)
            else:
                for batch_X, batch_y in zip(X.iter_batches(batch_size = self.batch_size), y.iter_batches(batch_size = self.batch_size)):
                    self.clf.partial_fit(batch_X, batch_y)

        dump(self.clf, self.clf_file)

    def predict(self, df, threshold = 0.8):
        if self.classifier in ['onesvm','linearsvm']:
            y_pred = _predict_binary(df)
        elif self.classifier in ['sgd','svm','mlr','mnb']:
            y_pred = _predict_multi(df, threshold)

        return y_pred

    def _predict_binary(self, df):
        y_pred = np.empty(nb_ids, dtype=np.int32)

        for i, row in enumerate(df.iter_rows()):
            y_pred[i] = clf.predict(row)

        return _label_decode(y_pred)

    def _predict_multi(self, df, threshold):
        y_pred = []

        for i, row in enumerate(df.iter_rows()):
            predicted = clf.predict_proba(row)
            if predicted[0,np.argmax(predict[0])] >= threshold:
                y_pred.append(self.labels_list[np.argmax(predicted[0])])
            else:
                y_pred.append(-1)

        return _label_decode(y_pred)

class KerasTFModel(ModelsUtils):
    '''
    Class to be used to build, train and predict models using Ray with Keras Tensorflow backend
    '''
    def __init__(classifier, dataset, outdir_model, outdir_results, batch_size, k, verbose):
        super().__init__(classifier, dataset, outdir_model, outdir_results, batch_size, k, verbose)
        # Parameters
        self.clf_file = '{}bacteria_binary_classifier_K{}_{}_{}_model'.format(outdir_model, k, classifier, dataset)
        # # Initialize empty
        self.nb_classes = None
        # Variables for training with Ray
        self.tf_config = json.loads(os.environ['TF_CONFIG'])
        self.num_workers = len(self.tf_config['cluster']['worker'])
        self.global_batch_size = self.batch_size * self.num_workers
        self.strategy = tf.distribute.MultiWorkerMirroredStrategy()
        if len(self.tf_config['GPU']) > 0:
            self.trainer = Trainer(backend = 'tensorflow', num_workers = self.num_workers, use_gpu = True)
        else:
            self.trainer = Trainer(backend = 'tensorflow', num_workers = self.num_workers)

    def _build(self):
        with self.strategy.scope():
            if self.classifier == 'attention':
                if self.verbose:
                    print('Training bacterial / host classifier based on Attention Weighted Neural Network')
                self.clf = build_attention(self.k)
            elif self.classifier == 'lstm':
                if self.verbose:
                    print('Training bacterial / host classifier based on Shallow LSTM Neural Network')
                self.clf = build_LSTM(self.k, self.batch_size)
            elif self.classifier == 'deeplstm':
                if self.verbose:
                    print('Training bacterial / host classifier based on Deep LSTM Neural Network')
                self.clf = build_deepLSTM(self.k, self.batch_size)
            elif self.classifier == 'lstm_attention':
                if self.verbose:
                    print('Training multiclass classifier based on Deep Neural Network hybrid between LSTM and Attention')
                self.clf = build_LSTM_attention(self.k, self.nb_classes, self.batch_size)
            elif self.classifier == 'cnn':
                if self.verbose:
                    print('Training multiclass classifier based on CNN Neural Network')
                self.clf = build_CNN(self.k, self.batch_size, self.nb_classes)
            elif self.classifier == 'widecnn':
                if self.verbose:
                    print('Training multiclass classifier based on Wide CNN Network')
                self.clf = build_wideCNN(self.k, self.batch_size, self.nb_classes)

    def _fit_model(self, X, y):
        X = self.scaleX(X)
        y = _label_encode(y)
        self.nb_classes = len(self.labels_list)
        self.multi_worker_dataset = self._join_shuffle_data(X, y, self.global_batch_size)

        checkpoint_strategy = CheckpointStrategy(num_to_keep = 1,
                                    checkpoint_score_attribute='val_accuracy',
                                    checkpoint_score_order='max')

        self.trainer.start()
        self.trainer.run(self._fit_ray, checkpoint_strategy = checkpoint_strategy)
        self.checkpoint = self.trainer.best_checkpoint
        self.trainer.shutdown()

    def _fit_ray(self):

        self._build()

        early = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)

        self.clf.fit(self.multi_worker_dataset, epochs = self.training_epochs)

        save_checkpoint(model_weights = self.clf.get_weights())

    def _join_shuffle_data(X_train, y_train, batch_size):
        # Join
        X_train = X_train.to_modin()
        y_train = y_train.to_modin()
        df = ray.data.from_modin(X_train.merge(y_train, on = 'id', how = 'left'))
        df = df.random_shuffle()
        df = df.to_tf(
        label_column = 'classes',
        batch_size = batch_size,
        output_signature = (
            tf.TensorSpec(shape=(None, batch_size), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),))

        return df

    def predict(self, df, threshold = 0.8):
        if self.classifier in ['attention','lstm','deeplstm']:
            y_pred = _predict_binary(df)
        elif self.classifier in ['lstm_attention','cnn','widecnn']:
            y_pred = _predict_multi(df, threshold)

        return y_pred

    def _predict_binary(self, df):
        predictor = TensorflowPredictor().from_checkpoint(checkpoint = self.checkpoint, model_definition = self._build)

        predicted = np.array(predictor.predict(df))

        y_pred = np.around(predicted.reshape(1, predicted.size)[0]).astype(np.int64)

        return _label_decode(y_pred)

    def _predict_multi(self, df, threshold):
        y_pred = []
        predictor = TensorflowPredictor().from_checkpoint(checkpoint = self.checkpoint, model_definition = self._build)

        predicted = np.array(predictor.predict(df))

        for i in range(len(predicted)):
            if np.argmax(predicted[i]) >= threshold:
                y_pred.append(self.labels_list[np.argmax(predicted[i])])
            else:
                y_pred.append(-1)

        return _label_decode(y_pred)
