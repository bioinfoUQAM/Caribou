import numpy as np

import os

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDOneClassSVM, SGDClassifier

from ray.util.joblib import register_ray
from joblib import parallel_backend, dump, load

from models.models_utils import Models_utils

__author__ = 'Nicolas de Montigny'

__all__ = []

'''
Class to be used to build, train and predict models using Ray with Scikit-learn backend
'''

class Sklearn_model(Models_utils):
    def __init__(classifier, dataset, outdir_model, outdir_results, batch_size, k, verbose):
        # Parameters
        self.classifier = classifier
        self.clf_file = '{}bacteria_binary_classifier_K{}_{}_{}_model.jb'.format(outdir_model, k, classifier, dataset)
        self.outdir = outdir_results
        self.batch_size = batch_size
        self.k = k
        self.verbose = verbose
        # Computes
        self.clf = self._build()


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

    def train(self, X, y, cv = True):
        if cv:
            self._cross_validation(X, y)
        else:
            self._fit_model(X, y)

    def _fit_model(self, X, y):
        X = self.scaleX(X)
        y = _label_encode(y)
        self.labels_list = np.unique(y['classes'])
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
