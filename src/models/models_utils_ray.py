import numpy as np
import modin.pandas as pd
import matplotlib.pyplot as plt

import re
import os
import ray
import glob
import shutil
import warnings

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from keras.callbacks import EarlyStopping
from tensorflow.keras.models import clone_model

from ray.util.joblib import register_ray
from joblib import Parallel, delayed, parallel_backend, dump, load


__author__ = 'Nicolas de Montigny'

__all__ = ['scaleX','cv_score','make_score_df','choose_delete_model','plot_figure',
           'cross_validation_training','fit_predict_cv','fit_model','model_predict',
           'fit_model_oneSVM_sk','fit_model_sk','fit_model_keras']

# Ignore warnings to have a more comprehensible output on stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

register_ray()

# Utils for all types of models / framework
################################################################################

# Data scaling
def scaleX(df):
    kmers = list(df.columns)
    kmers.remove('id')
    kmers.remove('classes')
    with parallel_backend('ray'):
        scaler = StandardScaler()
        df[kmers] = scaler.fit_transform(df[kmers])

    return df

# Choose model that performed better for keeping
def choose_delete_model(df_scores):
    clf_max = df_scores.idxmax()[2]
    path = os.path.dirname(clf_max)
    models_list = glob.glob(os.path.join(path, re.sub('_iter_\d+..json','',clf_max)) + '*')
    models_list.remove(clf_max)
    for file in models_list:
        os.remove(file)
    clf = re.sub('_iter_\d+', '',clf_max)
    os.rename(clf_max, clf)

    return clf

# Outputs scores for cross validation in a dictionnary
def cv_score(y_true, y_pred, classifier):
    scores = {clf_name:{}}

    if classifier in ['onesvm','linearsvm', 'attention','lstm','deeplstm']:
        average = 'binary'
    elif classifier in ['sgd','svm','mlr','mnb','lstm_attention','cnn','widecnn']:
        average = 'macro'

    support = precision_recall_fscore_support(y_true, y_pred , average = average)

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
def cross_validation_training(X_train, y_train, batch_size, k, classifier, outdir_plots, clf, training_epochs, cv = 1, shuffle = True, verbose = True, clf_file = None, n_jobs = 1):

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
                                (X_train, y_train, batch_size, classifier, clone(clf),
                                training_epochs, shuffle = shuffle, clf_file = clf_name)
                                for clf_name, df in zip(clf_names, df_data))

    elif classifier in ['attention','lstm','deeplstm','lstm_attention','cnn','widecnn']:
        with parallel_backend('threading'):
            cv_scores = Parallel(n_jobs = -1, prefer = 'threads', verbose = 100 if verbose else 0)(
                                delayed(fit_predict_cv)
                                (X_train, y_train, batch_size, classifier, clone_model(clf),
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
def fit_predict_cv(X_train, y_train, batch_size, classifier, clf, training_epochs, shuffle = True, clf_file = None):
    X_train = scaleX(X_train)
    cls = np.unique(df['classes'])

    with parallel_backend('ray'):
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size = 0.8, random_state=42)

    if classifier in ['onesvm','linearsvm','sgd','svm','mlr','mnb']:
        if classifier == 'onesvm':
            fit_model_oneSVM_sk(clf, X_train, batch_size, shuffle, clf_file)
        else:
            fit_model_sk(clf, X_train, y_train, cls, batch_size, shuffle, clf_file, predict_type = 'predict' if classifier == 'linearsvm' else 'predict_proba')

    elif classifier in ['attention','lstm','deeplstm','lstm_attention','cnn','widecnn']:
        fit_model_keras(clf, X_train, y_train, training_epochs, clf_file)

    y_pred = model_predict(X_test, clf_file)
    return cv_score(y_test, y_pred, classifier)

def fit_model(X_train, y_train, batch_size, classifier, clf, training_epochs, shuffle = True, clf_file = None):
    X_train = scaleX(X_train)
    cls = np.unique(df['classes'])

    if classifier in ['onesvm','linearsvm','sgd','svm','mlr','mnb']:
        if classifier == 'onesvm':
            clf = fit_model_oneSVM_sk(clf, X_train, batch_size, shuffle, clf_file)
        else:
            clf = fit_model_sk(clf, X_train, y_train, cls, batch_size, shuffle, clf_file, predict_type = 'predict' if classifier == 'linearsvm' else 'predict_proba')

    elif classifier in ['attention','lstm','deeplstm','lstm_attention','cnn','widecnn']:
        clf = fit_model_keras(clf, X_train, y_train, training_epochs, clf_file)

"""
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

def model_predict(clf_file, X, kmers_list, ids, classifier, nb_classes, labels_list, threshold = 0.8, verbose = True):
    predict = []
    y = pd.Series(range(len(ids)))

    if classifier in ["onesvm","linearsvm"]:
        generator = iter_generator(X, y, 1, kmers_list, ids, classifier, cv = 0, shuffle = False, training = False)
        predict = predict_binary_sk(clf_file, len(ids), generator)
        generator.handle.close()
    elif classifier in ["attention","lstm","deeplstm"]:
        generator = iter_generator_keras(X, y, 1, kmers_list, ids, 0, classifier, shuffle = False, training = False)
        predict = predict_binary_keras(clf_file, generator)
        generator.handle.close()
    elif classifier in ["sgd","svm","mlr","mnb"]:
        generator = iter_generator(X, y, 1, kmers_list, ids, classifier, cv = 0, shuffle = False, training = False)
        predict = predict_multi_sk(clf_file, labels_list, generator, threshold = threshold)
        generator.handle.close()
    elif classifier in ["lstm_attention","cnn","widecnn"]:
        generator = iter_generator_keras(X, y, 1, kmers_list, ids, 0, classifier, shuffle = False, training = False)
        predict = predict_multi_keras(clf_file, labels_list, generator, threshold = threshold)
        generator.handle.close()

    return predict
"""
# Scikit-learn versions
################################################################################
def fit_model_oneSVM_sk(clf, X_train, clf_file):
# TEST TO SEE IF FIT WORKS / PARTIAL_FIT NEEDED
    with parallel_backend('ray'):
        clf.fit(X_train)

    dump(clf, clf_file)

def fit_model_sk(clf, X_train, y_train, clf_file):
# TEST TO SEE IF FIT WORKS / PARTIAL_FIT NEEDED
    with parallel_backend('ray'):
        clf.fit(X_train, y_train)

    dump(clf, clf_file)
"""
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
"""
# Keras versions
################################################################################
def fit_model_keras

def fit_model_keras(clf, train_generator, val_generator, training_epochs, clf_file):
    modelcheckpoint = ModelCheckpoint(filepath=clf_file,monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
    early = EarlyStopping(monitor='val_accuracy', mode='min', verbose=1, patience=10)
    clf.fit(x = train_generator,
            validation_data = val_generator,
            epochs = training_epochs,
            callbacks = [modelcheckpoint,early],
            use_multiprocessing = True,
            workers = os.cpu_count())

"""
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
"""
