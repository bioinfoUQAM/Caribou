
import pandas as pd
import numpy as np
import tables as tb
import matplotlib.pyplot as plt

import os
import sys
import shutil
import glob
import re

from sklearn.metrics import precision_recall_fscore_support
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from metagenomics_ML.data.generators import iter_generator, iter_generator_keras

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

from tensorflow.keras.models import clone_model

from joblib import dump, load, Parallel, delayed, wrap_non_picklable_objects

import warnings

__author__ = "Nicolas de Montigny"

__all__ = ['scaleX','test_labels','cv_score','make_score_df','choose_delete_models_sk','choose_delete_models_keras','plot_figure',
           'cross_validation_training','fit_predict_cv','fit_model','model_predict',
           'fit_model_oneSVM_sk','fit_model_linear_sk','fit_model_multi_sk','predict_binary_sk','predict_multi_sk',
           'fit_model_keras','predict_binary_keras','predict_multi_keras']

# Ignore warnings to have a more comprehensible output on stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Utils for all types of models / framework
################################################################################

# X data scaling
def scaleX(X_data, y_data, batch_size, kmers, ids, cv = 0, shuffle = False, verbose = True):
    try:
        scaler = StandardScaler()
        generator = iter_generator(X_data, y_data, batch_size, kmers, ids, None, cv = 0, shuffle = False, training = False)
        for i, (X, y) in enumerate(generator.iterator):
            scaler.partial_fit(X)
        generator.handle.close()
        with tb.open_file(X_data, "a") as handle:
            handle.root.scaled = handle.create_carray("/", "scaled", obj = np.array(scaler.transform(handle.root.data.read()),dtype=np.float32))
    except tb.exceptions.NodeError:
        if verbose:
            print("Data already scaled")

# Return labels of dataset based on generator used
def test_labels(generator):
    y_test = np.empty(len(generator.positions_list), dtype = np.int32)
    for i in range(len(generator.positions_list)):
        y_test[i] = generator.labels[generator.positions_list[i]]
    return y_test

# Outputs scores for cross validation in a dictionnary
def cv_score(y_pred_test, y_test, clf_name, labels_list):
    scores = {clf_name:{}}

    if len(labels_list) == 2:
        y_test = y_test.astype(np.int64)

    support = precision_recall_fscore_support(y_test, y_pred_test, average = 'weighted')

    scores[clf_name]["Precision"] = support[0]
    scores[clf_name]["Recall"] = support[1]
    scores[clf_name]["F-score"] = support[2]

    return scores

def make_score_df(clf_scores):
    # rows are iterations
    # columns are score names
    df_scores = pd.DataFrame(columns = ["Precision", "Recall", "F-score"], index = clf_scores.keys(), dtype = np.float64)

    for i, clf_name in enumerate(clf_scores):
        df_scores.loc[clf_name,"Precision"] = clf_scores[clf_name]["Precision"]
        df_scores.loc[clf_name,"Recall"] = clf_scores[clf_name]["Recall"]
        df_scores.loc[clf_name,"F-score"] = clf_scores[clf_name]["F-score"]

    return df_scores

def choose_delete_models_keras(df_scores):
    clf_max = df_scores.idxmax()[2]
    path = os.path.dirname(clf_max)
    models_list = glob.glob(os.path.join(path, re.sub('_iter_\d+', '',clf_max)) + "*")
    models_list.remove(clf_max)
    for dir in models_list:
        shutil.rmtree(dir)

    return clf_max

def choose_delete_models_sk(df_scores):
    clf_max = df_scores.idxmax()[2]
    path = os.path.dirname(clf_max)
    models_list = glob.glob(os.path.join(path, re.sub('_iter_\d+..jb', '',clf_max)) + "*")
    models_list.remove(clf_max)
    for file in models_list:
        os.remove(file)

    return clf_max


# Outputs results and plots of multiple cross validation iterations
def plot_figure(df_scores, n_jobs, outdir_plots, k, classifier):
    if classifier in ["onesvm","linearsvm","attention","lstm","deeplstm"]:
        clf_type = "extraction"
    elif classifier in ["ridge","svm","mlr","mnb","lstm_attention","cnn","deepcnn"]:
        clf_type = "classification"

    plot_file = "{}_K{}_{}_{}_cv_{}.png".format(outdir_plots, k, clf_type, classifier, "metrics")
    x = np.arange(1, n_jobs + 1, dtype = np.int32)
    y_p = np.array(df_scores.loc[:, "Precision"])
    y_r = np.array(df_scores.loc[:, "Recall"])
    y_f = np.array(df_scores.loc[:, "F-score"])

    plt.plot(x, y_p, linestyle='-', marker='o', label = "Precision")
    plt.plot(x, y_r, linestyle='-', marker='o', label = "Recall")
    plt.plot(x, y_f, linestyle='-', marker='o', label = "F-score")
    plt.xlabel("Training iterations")
    plt.ylabel("Score")
    plt.legend(loc='upper left', fancybox=True, shadow=True,bbox_to_anchor=(1.01, 1.02))
    plt.suptitle("Cross-validation scores over {} training iterations using classifier {} for {} on kmers of length {}".format(n_jobs, clf_type, classifier, k))
    plt.savefig(plot_file, bbox_inches = "tight", format = "png", dpi = 150)

# Caller functions
################################################################################

# Model training with cross-validation
def cross_validation_training(X_train, y_train, batch_size, kmers, ids, classifier, labels_list, outdir_plots, clf, cv = 1, shuffle = True, threshold = 0.8, verbose = True, clf_file = None, n_jobs = 1):
    # cv_scores is a list of n fold dicts
    # each dict contains the results of the iteration
    cv_scores = []
    clf_scores = {}
    parallel = Parallel(n_jobs = n_jobs if n_cvJobs <= os.cpu_count() else -1, backend = 'loky', prefer = "processes", verbose = 100 if verbose else 0)

    if classifier in ["onesvm","linearsvm","ridge","svm","mlr","mnb"]:
        clf_file, ext = os.path.splitext(clf_file)
        clf_names = ["{}_iter_{}.{}".format(clf_file, iter, ext) for iter in range(n_jobs)]
        X_data = ["{}_iter_{}".format(X_train, iter) for iter in range(n_jobs)]
        for file in X_data:
            shutil.copy(X_train,file)
        cv_scores = parallel(delayed(fit_predict_cv)
            (X_file, y_train, batch_size, kmers,
            ids, classifier, labels_list, outdir_plots,
            clone(clf), cv = 1, shuffle = True,
            threshold = threshold, verbose = True, clf_file = clf_name)
            for clf_name, X_file in zip(clf_names,X_data))
        for file in X_data:
            os.remove(file)
    elif classifier in ["attention","lstm","deeplstm","lstm_attention","cnn","deepcnn"]:
        clf_names = ["{}_iter_{}".format(clf_file, iter) for iter in range(n_jobs)]
        X_data = ["{}_iter_{}".format(X_train, iter) for iter in range(n_jobs)]
        for file in X_data:
            shutil.copy(X_train,file)
        cv_scores = parallel(delayed(fit_predict_cv)
            (X_file, y_train, batch_size, kmers,
            ids, classifier, labels_list, outdir_plots,
            clone_model(clf), cv = 1, shuffle = True,
            threshold = threshold, verbose = True, clf_file = clf_name)
            for clf_name, X_file in zip(clf_names,X_data))
        for file in X_data:
            os.remove(file)
    # Transform list of dictionnaries into one dictionnary
    for i, clf in enumerate(clf_names):
        clf_scores[clf] = cv_scores[i][clf]

    df_scores = make_score_df(clf_scores)

    plot_figure(df_scores, n_jobs, outdir_plots, len(kmers[1]), classifier)

    if classifier in ["onesvm","linearsvm","ridge","svm","mlr","mnb"]:
        clf_file = choose_delete_models_sk(df_scores)
    elif classifier in ["attention","lstm","deeplstm","lstm_attention","cnn","deepcnn"]:
        clf_file = choose_delete_models_keras(df_scores)

    return clf_file

@wrap_non_picklable_objects
def fit_predict_cv(X_train, y_train, batch_size, kmers, ids, classifier, labels_list, outdir_plots, clf, cv = 1, shuffle = True, threshold = 0.8, verbose = True, clf_file = None):
    scaleX(X_train, y_train, batch_size, kmers, ids, verbose)

    if classifier in ["onesvm","linearsvm"]:
        if classifier == "onesvm":
            train_generator, test_generator = iter_generator(X_train, y_train, batch_size, kmers, ids, classifier, cv = cv, shuffle = shuffle, training = True)
            fit_model_oneSVM_sk(clf, train_generator, clf_file)
            train_generator.handle.close()
        elif classifier == "linearsvm":
            train_generator, test_generator = iter_generator(X_train, y_train, batch_size, kmers, ids, classifier, cv = cv, shuffle = shuffle, training = True)
            fit_model_linear_sk(clf, train_generator, clf_file)
            train_generator.handle.close()
        y_pred_test = predict_binary_sk(clf_file, ids, test_generator)
        y_test = test_labels(test_generator)
        test_generator.handle.close()

    elif classifier in ["ridge","svm","mlr","mnb"]:
        train_generator, test_generator = iter_generator(X_train, y_train, batch_size, kmers, ids, classifier, cv = cv, shuffle = shuffle, training = True)
        fit_model_multi_sk(clf, train_generator, clf_file, cls = np.unique(y_train))
        train_generator.handle.close()
        y_pred_test = predict_multi_sk(clf_file, labels_list, test_generator, threshold = threshold)
        y_test = test_labels(test_generator)
        test_generator.handle.close()

    elif classifier in ["attention","lstm","deeplstm"]:
        train_generator, val_generator, test_generator = iter_generator_keras(X_train, y_train, batch_size, kmers, ids, cv, classifier, shuffle = shuffle, training = True)
        clf.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        fit_model_keras(clf, train_generator, val_generator, clf_file)
        train_generator.handle.close()
        val_generator.handle.close()
        y_pred_test = predict_binary_keras(clf_file, test_generator)
        y_test = test_labels(test_generator)
        test_generator.handle.close()

    elif classifier in ["lstm_attention","cnn","deepcnn"]:
        train_generator, val_generator, test_generator = iter_generator_keras(X_train, y_train, batch_size, kmers, ids, cv, classifier, shuffle = shuffle, training = True)
        clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        fit_model_keras(clf, train_generator, val_generator, clf_file)
        train_generator.handle.close()
        val_generator.handle.close()
        y_pred_test = predict_multi_keras(clf_file, labels_list, test_generator, threshold = threshold)
        y_test = test_labels(test_generator)
        test_generator.handle.close()

    return cv_score(y_pred_test, y_test, clf_file, labels_list)

def fit_model(X_train, y_train, batch_size, kmers, ids, classifier, labels_list, clf, cv = 0, shuffle = True, verbose = True, clf_file = None):
    scaleX(X_train, y_train, batch_size, kmers, ids, verbose)

    if classifier == "onesvm":
        generator = iter_generator(X_train, y_train, batch_size, kmers, classifier, cv = cv, shuffle = shuffle, training = True)
        fit_model_oneSVM_sk(clf, generator, clf_file)
        generator.handle.close()
    elif classifier == "linearsvm":
        generator = iter_generator(X_train, y_train, batch_size, kmers, classifier, cv = cv, shuffle = shuffle, training = True)
        fit_model_linear_sk(clf, generator, clf_file)
        generator.handle.close()
    elif classifier in ["ridge","svm","mlr","mnb"]:
        generator = iter_generator(X_train, y_train, batch_size, kmers, classifier, cv = cv, shuffle = shuffle, training = True)
        fit_model_multi_sk(clf, generator, clf_file, labels_list, cls = np.unique(y_train))
        generator.handle.close()
    elif classifier in ["attention","lstm","deeplstm","lstm_attention","cnn","deepcnn"]:
        train_generator, val_generator = iter_generator_keras(X_train, y_train, batch_size, kmers, ids, cv, classifier, shuffle = shuffle, training = True)
        fit_model_keras(clf, train_generator, val_generator, clf_file)
        train_generator.handle.close()
        val_generator.handle.close()

def model_predict(clf_file, X, kmers_list, ids, classifier, nb_classes, labels_list, threshold = 0.8, verbose = True):
    y = pd.Series(range(len(ids)))
    scaleX(X, y, 32, kmers_list, ids, verbose)

    if classifier in ["onesvm","linearsvm"]:
        generator = iter_generator(X, y, 1, kmers_list, ids, classifier, cv = 0, shuffle = False, training = False)
        predict = predict_binary_sk(clf_file, ids, generator)
        generator.handle.close()
    elif classifier in ["attention","lstm","deeplstm"]:
        generator = iter_generator_keras(X, y, 1, kmers_list, ids, 0, classifier, shuffle = False, training = False)
        predict = predict_binary_keras(clf_file, generator)
        generator.handle.close()
    elif classifier in ["ridge","svm","mlr","mnb"]:
        generator = iter_generator(X, y, 1, kmers_list, ids, classifier, cv = 0, shuffle = False, training = False)
        predict = predict_multi_sk(clf_file, labels_list, generator, threshold = threshold)
        generator.handle.close()
    elif classifier in ["lstm_attention","cnn","deepcnn"]:
        generator = iter_generator_keras(X, y, 1, kmers_list, ids, 0, classifier, shuffle = False, training = False)
        predict = predict_multi_keras(clf_file, labels_list, generator, threshold = threshold)
        generator.handle.close()

    return predict

# Scikit-learn versions
################################################################################
def fit_model_oneSVM_sk(clf, generator, clf_file):
    for i, (X, y) in enumerate(generator.iterator):
        clf.partial_fit(X, y)
    dump(clf, clf_file)

def fit_model_linear_sk(clf, generator, clf_file):
    for i, (X, y) in enumerate(generator.iterator):
        clf.partial_fit(X, y, classes = np.array([-1,0,1], dtype = float))
    dump(clf, clf_file)

def fit_model_multi_sk(clf, generator, clf_file, cls):
    for i, (X, y) in enumerate(generator.iterator):
        clf.partial_fit(X, y, classes = cls)
    clf = CalibratedClassifierCV(base_estimator = clf, cv = "prefit").fit(X,y)
    dump(clf, clf_file)

def predict_binary_sk(clf_file, ids, generator):
    y_pred = np.empty(len(ids), dtype=np.int32)

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

# Keras versions
################################################################################
def fit_model_keras(clf, train_generator, val_generator, clf_file):
    modelcheckpoint = ModelCheckpoint(filepath=clf_file,monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
    early = EarlyStopping(monitor='val_accuracy', mode='min', verbose=1, patience=10)
    clf.fit(x = train_generator,
            validation_data = val_generator,
            epochs = 100,
            callbacks = [modelcheckpoint,early],
            use_multiprocessing = True,
            workers = os.cpu_count())

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
