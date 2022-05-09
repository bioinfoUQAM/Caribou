import pandas as pd
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

__all__ = ['scaleX','test_labels','cv_score','make_score_df','choose_delete_models_sk','choose_delete_models_keras','plot_figure',
           'cross_validation_training','fit_predict_cv','fit_model','model_predict',
           'fit_model_oneSVM_sk','fit_model_linear_sk','fit_model_multi_sk','predict_binary_sk','predict_multi_sk',
           'fit_model_keras','predict_binary_keras','predict_multi_keras']

# Ignore warnings to have a more comprehensible output on stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Utils for all types of models / framework
################################################################################

# Data scaling
def scaleX(df, kmers):
    scaler = vaex.ml.StandardScaler(features = kmers, prefix = 'scaled_')
    scaler.fit(df)
    df = scaler.transform(df)

    return df

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

# Outputs scores for cross validation in a dictionnary
def cv_score(df, classifier):
    scores = {clf_name:{}}

    if classifier in ['onesvm','linearsvm', 'attention','lstm','deeplstm']:
        average = 'binary'
    elif classifier in ['sgd','svm','mlr','mnb','lstm_attention','cnn','widecnn']:
        average = 'macro'

    support = df.ml.metrics.precision_recall_fscore(df.label_encoded_classes, df.predicted_classes , average = average)

    scores[clf_name]['Precision'] = support[0]
    scores[clf_name]['Recall'] = support[1]
    scores[clf_name]['F-score'] = support[2]

    return scores

def make_score_df(clf_scores):
    # rows are iterations
    # columns are score names
    df_scores = pd.DataFrame(columns = ['Precision', 'Recall', 'F-score'], index = clf_scores.keys(), dtype = np.float64)

    for i, clf_name in enumerate(clf_scores):
        df_scores.loc[clf_name,'Precision'] = clf_scores[clf_name]['Precision']
        df_scores.loc[clf_name,'Recall'] = clf_scores[clf_name]['Recall']
        df_scores.loc[clf_name,'F-score'] = clf_scores[clf_name]['F-score']

    return df_scores

# Outputs results and plots of multiple cross validation iterations
def plot_figure(df_scores, n_jobs, outdir_plots, k, classifier):
    if classifier in ['onesvm','linearsvm','attention','lstm','deeplstm']:
        clf_type = 'extraction'
    elif classifier in ['sgd','svm','mlr','mnb','lstm_attention','cnn','widecnn']:
        clf_type = 'classification'

    plot_file = '{}_K{}_{}_{}_cv_{}.png'.format(outdir_plots, k, clf_type, classifier, 'metrics')
    x = np.arange(1, n_jobs + 1, dtype = np.int32)
    y_p = np.array(df_scores.loc[:, 'Precision'])
    y_r = np.array(df_scores.loc[:, 'Recall'])
    y_f = np.array(df_scores.loc[:, 'F-score'])

    plt.plot(x, y_p, linestyle='-', marker='o', label = 'Precision')
    plt.plot(x, y_r, linestyle='-', marker='o', label = 'Recall')
    plt.plot(x, y_f, linestyle='-', marker='o', label = 'F-score')
    plt.xlabel('Training iterations')
    plt.ylabel('Score')
    plt.legend(loc='upper left', fancybox=True, shadow=True,bbox_to_anchor=(1.01, 1.02))
    plt.suptitle('Cross-validation scores over {} training iterations using classifier {} for {} on kmers of length {}'.format(n_jobs, clf_type, classifier, k))
    plt.savefig(plot_file, bbox_inches = 'tight', format = 'png', dpi = 150)

# Caller functions
################################################################################

# Multiple parallel model training with cross-validation
def cross_validation_training(df, batch_size, k, classifier, outdir_plots, clf, training_epochs, cv = 1, shuffle = True, verbose = True, clf_file = None, n_jobs = 1):

    # cv_scores is a list of n fold dicts
    # each dict contains the results of the iteration
    cv_scores = []
    clf_scores = {}

    clf_file, ext = os.path.splitext(clf_file)
    clf_names = ['{}_iter_{}{}'.format(clf_file, iter, ext) for iter in range(n_jobs)]
    df_file, ext = os.path.splitext(df)
    df_data = ['{}_iter_{}{}'.format(df_file, iter, ext) for iter in range(n_jobs)]

    for file in df_data:
        shutil.copy(df, file)

    if classifier in ['onesvm','linearsvm','sgd','svm','mlr','mnb']:
        with parallel_backend('threading'):
            cv_scores = Parallel(n_jobs = -1, prefer = 'threads', verbose = 100 if verbose else 0)(
                                delayed(fit_predict_cv)
                                (df, batch_size, classifier, clone(clf),
                                training_epochs, shuffle = shuffle, clf_file = clf_name)
                                for clf_name, df in zip(clf_names, df_data))

    elif classifier in ['attention','lstm','deeplstm','lstm_attention','cnn','widecnn']:
        with parallel_backend('threading'):
            cv_scores = Parallel(n_jobs = -1, prefer = 'threads', verbose = 100 if verbose else 0)(
                                delayed(fit_predict_cv)
                                (df, batch_size, classifier, clone_model(clf),
                                training_epochs, shuffle = shuffle, clf_file = clf_name)
                                for clf_name, df in zip(clf_names, df_data))
    for file in X_data:
        os.remove(file)

    # Transform list of dictionnaries into one dictionnary
    for i, clf in enumerate(clf_names):
        clf_scores[clf] = cv_scores[i][clf]

    df_scores = make_score_df(clf_scores)

    plot_figure(df_scores, n_jobs, outdir_plots, k, classifier)

    clf_file = choose_delete_model(df_scores)

    return clf_file

# Model training and cross validating individually
def fit_predict_cv(df, batch_size, classifier, clf, training_epochs, shuffle = True, clf_file = None):
    scaleX(df)
    cls = df.unique('label_encoded_classes')

    if classifier in ['onesvm','linearsvm','sgd','svm','mlr','mnb']:
        df_train, df_test = df.split_random([0.8, 0.2], random_state=42)
        if classifier == 'onesvm':
            fit_model_oneSVM_sk(clf, df_train, batch_size, shuffle, clf_file)
        else:
            fit_model_sk(clf, df, cls, batch_size, shuffle, clf_file, predict_type = 'predict' if classifier == 'linearsvm' else 'predict_proba')

    elif classifier in ['attention','lstm','deeplstm','lstm_attention','cnn','widecnn']:
        df_train, df_valid, df_test = df.split_random([0.8, 0.1, 0.1], random_state=42)
        fit_model_keras(clf, df_train, df_valid, training_epochs, clf_file)

    df_test = model_predict(df_test, clf_file)
    return cv_score(df_test, classifier)

def fit_model(df, batch_size, classifier, clf, training_epochs, shuffle = True, clf_file = None):
    scaleX(df)
    cls = df.unique('label_encoded_classes')

    if classifier in ['onesvm','linearsvm','sgd','svm','mlr','mnb']:
        if classifier == 'onesvm':
            fit_model_oneSVM_sk(clf, df_train, batch_size, shuffle, clf_file)
        else:
            fit_model_sk(clf, df, cls, batch_size, shuffle, clf_file, predict_type = 'predict' if classifier == 'linearsvm' else 'predict_proba')

    elif classifier in ['attention','lstm','deeplstm','lstm_attention','cnn','widecnn']:
        df_train, df_valid = df.split_random([0.8, 0.2], random_state=42)
        fit_model_keras(clf, df_train, df_valid, training_epochs, clf_file)

def model_predict(df, clf_file classifier, threshold = 0.8):
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
                                                 prediction_type = predict_type
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
