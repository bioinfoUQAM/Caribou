import numpy as np

import os

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDOneClassSVM, SGDClassifier

from ray.util.joblib import register_ray
from joblib import parallel_backend, dump, load

__author__ = 'Nicolas de Montigny'

__all__ = []

'''
Class to be used to build, train and predict models using Ray with Scikit-learn backend
'''

class Sklearn_model(Models_utils):
    def __init__(classifier, clf_file, outdir, batch_size, k, verbose):
        # Parameters
        self.classifier = classifier
        self.clf_file = clf_file
        self.outdir = outdir
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
        with parallel_backend('ray'):
            if self.classifier == 'onesvm':
                self.clf.fit(X)
            else:
                self.clf.fit(X, y)

        dump(self.clf, self.clf_file)

    def predict(self, X, df, clf_file, classifier, threshold = 0.8):
# TODO: Finish figuring how to predict with Ray
        print('To do')
        # predicted_prices = lm.predict(features)
        # return predict

################################################################################
'''
def predict_binary_sk(clf_file, nb_ids, generator):
    y_pred = np.empty(nb_ids, dtype=np.int32)

    clf = load(clf_file)
    for i, (X, y) in enumerate(generator.iterator):
        y_pred[i] = clf.predict(X)

    return y_pred

def predict_multi_sk(clf_file, labels_list, generator, threshold = 0.8):
    y_pred = []

    clf = load(clf_file)
    for i, (X, y) in enumerate(generator.iterator):
        predict = clf.predict_proba(X)
        if predict[0,np.argmax(predict[0])] >= threshold:
            y_pred.append(labels_list[np.argmax(predict[0])])
        else:
            y_pred.append(-1)

    return y_pred
'''
