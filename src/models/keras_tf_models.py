import numpy as np

import os
import json

from keras.callbacks import EarlyStopping

from ray import tune
from ray.train import Trainer
from ray.ml.predictors.tensorflow import TensorflowPredictor

from models.build_neural_networks import *

__author__ = 'Nicolas de Montigny'

__all__ = []

'''
Class to be used to build, train and predict models using Ray with Keras Tensorflow backend
'''
class Keras_TF_model(Models_utils):
    def __init__(classifier, clf_file, outdir, nb_classes, batch_size, k, verbose):
        # Parameters
        self.classifier = classifier
        self.clf_file = clf_file
        self.outdir = outdir
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


# TODO: ADAPT WITH CHECKPOINTS WITH RAY TRAIN/TUNE
    def train(self, X, y, cv = True):
        if cv:
            self._cross_validation(X, y)
        else:
            self._fit_model(X, y)

    def _fit_model(self, X, y):
        X = self.scaleX(X)
        self.multi_worker_dataset = self._join_shuffle_data(X, y, self.global_batch_size)

        self.trainer.start()
        self.trainer.run(self._fit_ray)
        self.trainer.shutdown()


    def _fit_ray(self):

        self._build()

# TODO: Checkpoints using RAY not keras
        modelcheckpoint = ModelCheckpoint(filepath=self.clf_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
        early = EarlyStopping(monitor='val_accuracy', mode='min', verbose=1, patience=10)

        self.clf.fit(self.multi_worker_dataset, epochs = self.training_epochs)

    def _join_shuffle_data(X_train, y_train, batch_size):
        df = ray.data.from_modin(X_train.join(y_train, on = 'id', how = 'left'))
        df = df.random_shuffle()
        df = df.to_tf(
        label_column = 'classes',
        batch_size = batch_size,
        output_signature = (
            tf.TensorSpec(shape=(None, batch_size), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),))

        return df

    def predict(self, X, df, clf_file, classifier, threshold = 0.8):
# TODO: Finish figuring how to predict with Ray
        print('To do')
        # predict = df.map_batches(load_model_keras)
        # return predict

####################################################

'''
def predict_keras():
    predictor = predictor = TensorflowPredictor()

def predict_binary_keras(clf_file, generator):
    clf = load_model(clf_file)
    predict = clf.predict(generator,
                          use_multiprocessing = True,
                          workers = os.cpu_count())

    y_pred = np.around(predict.reshape(1, predict.size)[0]).astype(np.int64)
    generator.handle.close()

    return y_pred

def predict_multi_keras(clf_file, labels_list, generator, threshold = 0.8):
    y_pred = []

    clf = load_model(clf_file)
    predict = clf.predict(generator,
                          use_multiprocessing = True,
                          workers = os.cpu_count())
    for i in range(len(predict)):
        if np.argmax(predict[i]) >= threshold:
            y_pred.append(labels_list[np.argmax(predict[i])])
        else:
            y_pred.append(-1)

    return y_pred
'''
