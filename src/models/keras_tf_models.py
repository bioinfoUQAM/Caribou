import numpy as np

import os
import ray
import json

from keras.callbacks import EarlyStopping

from ray.ml.predictors.tensorflow import TensorflowPredictor
from ray.train import Trainer, save_checkpoint, CheckpointStrategy

from models.build_neural_networks import *
from models.models_utils import Models_utils

__author__ = 'Nicolas de Montigny'

__all__ = []

'''
Class to be used to build, train and predict models using Ray with Keras Tensorflow backend
'''
class Keras_TF_model(Models_utils):
    def __init__(classifier, dataset, outdir_model, outdir_results, batch_size, k, verbose):
        # Parameters
        self.classifier = classifier
        self.clf_file = '{}bacteria_binary_classifier_K{}_{}_{}_model.jb'.format(outdir_model, k, classifier, dataset)
        self.outdir = outdir_results
        self.batch_size = batch_size
        self.k = k
        self.verbose = verbose
        self.nb_classes = nb_classes
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

    def train(self, X, y, cv = True):
        if cv:
            self._cross_validation(X, y)
        else:
            self._fit_model(X, y)

    def _fit_model(self, X, y):
        X = self.scaleX(X)
        y = _label_encode(y)
        self.labels_list = np.unique(y['classes'])
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
