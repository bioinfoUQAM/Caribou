import numpy as np
import modin.pandas as pd
import matplotlib.pyplot as plt

import re
import os
import ray
import glob
import json
import shutil
import warnings

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from keras.callbacks import EarlyStopping
from tensorflow.keras.models import clone_model

from ray import tune
from ray.train import Trainer
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
        if len(tf.config.list_physical_devices('GPU')) > 0:
            trainer = Trainer(backend = 'tensorflow', num_workers = num_workers, use_gpu = True)
        else:
            trainer = Trainer(backend = 'tensorflow', num_workers = num_workers)

            trainer.start()
            trainer.run(fit_model_keras, config = {'clf':clf, 'X_train':X_train, 'y_train':y_train, 'training_epochs':training_epochs, 'clf_file':clf_file})
            trainer.shutdown()

    y_pred = model_predict(X_test, clf_file)
    return cv_score(y_test, y_pred, classifier)

def fit_model(X_train, y_train, batch_size, classifier, clf, training_epochs, shuffle = True, clf_file = None):
    X_train = scaleX(X_train)
    cls = np.unique(df['classes'])

    if classifier in ['onesvm','linearsvm','sgd','svm','mlr','mnb']:
        if classifier == 'onesvm':
            fit_model_oneSVM_sk(clf, X_train, batch_size, shuffle, clf_file)
        else:
            fit_model_sk(clf, X_train, y_train, cls, batch_size, shuffle, clf_file, predict_type = 'predict' if classifier == 'linearsvm' else 'predict_proba')

    elif classifier in ['attention','lstm','deeplstm','lstm_attention','cnn','widecnn']:
        if len(tf.config.list_physical_devices('GPU')) > 0:
            trainer = Trainer(backend = 'tensorflow', num_workers = num_workers, use_gpu = True)
        else:
            trainer = Trainer(backend = 'tensorflow', num_workers = num_workers)

        trainer.start()
        trainer.run(fit_model_keras, config = {'clf':clf, 'X_train':X_train, 'y_train':y_train, 'training_epochs':training_epochs, 'clf_file':clf_file})
        trainer.shutdown()

"""
def model_predict(df, clf_file, classifier, threshold = 0.8):

    if classifier in ['onesvm','linearsvm','sgd','svm','mlr','mnb']:
        predict = df.map_batches(load_model_sk, config = {})
    elif classifier in ['attention','lstm','deeplstm','lstm_attention','cnn','widecnn']:
        predict = df.map_batches(load_model_keras)

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
# Keras versions
################################################################################
def join_shuffle_data(X_train, y_train, batch_size):
    df = ray.data.from_modin(X_train.join(y_train, on = 'id', how = 'left'))
    df = df.random_shuffle()
    df = df.to_tf(
        label_column = 'classes',
        batch_size = batch_size,
        output_signature = (
            tf.TensorSpec(shape=(None, batch_size), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),))

    return df

def fit_model_keras(config):
    # Environment variable setted by Ray
    tf_config = json.loads(os.environ['TF_CONFIG'])
    num_workers = len(tf_config['cluster']['worker'])

    global_batch_size = config['batch_size'] * num_workers
    multi_worker_dataset = join_shuffle_data(config['X_train'], config['y_train'], global_batch_size)

    modelcheckpoint = ModelCheckpoint(filepath=clf_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
    early = EarlyStopping(monitor='val_accuracy', mode='min', verbose=1, patience=10)
    config['clf'].fit(multi_worker_dataset,
            epochs = config['training_epochs'],
            callbacks = [modelcheckpoint, early])

'''
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
