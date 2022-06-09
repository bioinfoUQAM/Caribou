import modin.pandas as pd
import numpy as np
import tables as tb
import matplotlib.pyplot as plt

import re
import os
import vaex
import glob
import shutil
import warnings

from sklearn.base import clone
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import clone_model
from joblib import Parallel, delayed, parallel_backend


__author__ = 'Nicolas de Montigny'

__all__ = ['scaleX','cv_score','make_score_df','choose_delete_model','plot_figure',
           'cross_validation_training','fit_predict_cv','fit_model','model_predict',
           'fit_model_oneSVM_sk','fit_model_sk','fit_model_keras']

# Ignore warnings to have a more comprehensible output on stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Utils for all types of models / framework
################################################################################

# Choose model that performed better for keeping
def choose_delete_model(df_scores):
    clf_max = df_scores.idxmax()[2]
    path = os.path.dirname(clf_max)
    models_list = glob.glob(os.path.join(path, re.sub('_iter_\d+..json', '',clf_max)) + '*')
    models_list.remove(clf_max)
    for file in models_list:
        os.remove(file)
    clf = re.sub('_iter_\d+', '',clf_max)
    os.rename(clf_max, clf)

    return clf

# Caller functions
################################################################################

def model_predict(df, clf_file, classifier, threshold = 0.8):
    df.state_load(clf_file)

    if classifier in ['attention','lstm','deeplstm']:
        df['predicted_classes'] = np.around(df.predicted_classes.astype('int'))
    elif classifier in ['lstm_attention','cnn','widecnn','sgd','svm','mlr','mnb']:
        array_predicted = np.zeros(len(df))
        for i, predict in enumerate(df.predicted_classes.values):
            pos_argmax = np.argmax(predict)
            if predict[pos_argmax] >= threshold:
                array_predicted[i] = pos_argmax
            else:
                array_predicted[i] = -1
        df['predicted_classes'] = array_predicted

    return df

# Scikit-learn versions
################################################################################
def fit_model_oneSVM_sk(clf, df, batch_size, shuffle, clf_file):
    model = vaex.ml.sklearn.IncrementalPredictor(model = clf,
                                                 features = df.get_column_names(regex='^standard'),
                                                 target = 'label_encoded_classes',
                                                 batch_size = batch_size,
                                                 shuffle = shuffle,
                                                 prediction_name = 'predicted_classes')
    model.fit(df = df)
    df = model.transform(df)
    df.state_write(clf_file)

def fit_model_sk(clf, df, cls, batch_size, shuffle, clf_file, predict_type):
    model = vaex.ml.sklearn.IncrementalPredictor(model = clf,
                                                 features = df.get_column_names(regex='^standard'),
                                                 target = 'label_encoded_classes',
                                                 batch_size = batch_size,
                                                 num_epochs = training_epochs,
                                                 shuffle = shuffle,
                                                 prediction_type = predict_type,
                                                 prediction_name = 'predicted_classes',
                                                 partial_fit_kwargs = {'classes':cls})
    model.fit(df = df)
    df = model.transform(df)
    df.state_write(clf_file)

# Keras versions
################################################################################
def fit_model_keras(clf, df_train, df_valid, training_epochs, clf_file):
    features = df_valid.get_column_names(regex='^standard')
    train_generator = df_train.ml.tensorflow.to_keras_generator(features = features,
                                                                target = 'label_encoded_classes',
                                                                batch_size = batch_size)
    val_generator = df_valid.ml.tensorflow.to_keras_generator(features = features,
                                                                target = 'label_encoded_classes',
                                                                batch_size = batch_size)

    early = EarlyStopping(monitor='val_accuracy', mode='min', verbose=1, patience=10)
    clf.fit(x = train_generator,
            validation_data = val_generator,
            epochs = training_epochs,
            callbacks = [early],
            use_multiprocessing = True,
            workers = os.cpu_count())
    model = vaex.ml.tensorflow.KerasModel(model = clf,
                                          features = features,
                                          prediction_name = 'predicted_classes')
    df = model.transform(df_train)
    df_train.state_write(clf_file)
