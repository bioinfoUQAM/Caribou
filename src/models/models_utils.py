import modin.pandas as pd

from abc import ABC, abstractmethod

import os
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from joblib import parallel_backend
from ray.util.joblib import register_ray

__author__ = 'Nicolas de Montigny'

__all__ = []

# Ignore warnings to have a more comprehensible output on stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

register_ray()

'''
Utilities for both types of framework
'''

class Models_utils(ABC):
    # Data scaling
    def _scaleX(df):
        kmers = list(df.columns)
        kmers.remove('id')
        kmers.remove('classes')
        with parallel_backend('ray'):
            scaler = StandardScaler()
            df[kmers] = scaler.fit_transform(df[kmers])

        return df

    # Outputs scores for cross validation in a dictionnary
    def _cv_score(y_true, y_pred, classifier, k, outdir):

        if classifier in ['onesvm','linearsvm', 'attention','lstm','deeplstm']:
            average = 'binary'
        elif classifier in ['sgd','svm','mlr','mnb','lstm_attention','cnn','widecnn']:
            average = 'macro'

        support = precision_recall_fscore_support(y_true, y_pred , average = average)


        scores = pd.DataFrame({'Classifier':self.classifier,'Precision':support[0],'Recall':support[1],'F-score':support[2]})

        scores.to_csv(os.join(self.outdir,'{}_K{}_cv_scores.csv'.format(self.classifier, self.k)))

    def _cross_validation(self, X_train, y_train):
        with parallel_backend('ray'):
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size = 0.8, random_state=42)

        self._fit_model(X_train, y_train)

        y_pred = self.predict(X_test, clf_file)
        self._cv_score(y_test, y_pred)

    @abstractmethod
    def _fit_model(self):
        """
        """
