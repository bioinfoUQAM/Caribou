
import pandas as pd
import numpy as np
import tables as tb

import os
import sys

from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from data.generators import iter_generator, iter_generator_keras

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

from joblib import dump, load

import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Utils for all types of models / framework
################################################################################

# X data scaling
def scaleX(X_data, y_data, batch_size, kmers, ids, cv = 0, shuffle = False, verbose = True):
    try:
        scaler = StandardScaler()
        generator = iter_generator(X_data, y_data, batch_size, kmers, ids, cv = 0, shuffle = False, training = False)
        for i, (X, y) in enumerate(generator.iterator):
            scaler.partial_fit(X)
        generator.handle.close()
        with tb.open_file(X_data, "a") as handle:
            handle.root.scaled = handle.create_carray("/", "scaled", obj = np.array(scaler.transform(handle.root.data.read()),dtype=np.float32))
    except tb.exceptions.NodeError:
        if verbose:
            print("Data already scaled")

def test_labels(generator):
    y_test = np.zeros(len(generator.positions_list), dtype = int)
    for i in range(len(generator.positions_list)):
            y_test[i] = generator.labels[generator.positions_list[i]]
    return y_test

# Model training cross-validation
def training_cross_validation(y_pred_test, y_test, classifier):
    print("Cross validating classifier : " + str(classifier))
    print("Confidence matrix : ")
    print(metrics.confusion_matrix(y_pred_test, y_test))
    print("Precision : {}".format(str(metrics.precision_score(y_pred_test, y_test, average=None))))
    print("Recall : {}".format(str(metrics.recall_score(y_pred_test, y_test, average=None))))
    print("F-score : {}".format(str(metrics.f1_score(y_pred_test, y_test, average=None))))

# Caller functions
################################################################################

def fit_predict_cv(X_train, y_train, batch_size, kmers, ids, classifier, labels_list, clf, cv = 1, shuffle = True, verbose = True, clf_file = None):
    scaleX(X_train, y_train, batch_size, kmers, ids, verbose)

    if classifier in ["onesvm","linearsvm"]:
        train_generator, test_generator = iter_generator(X_train, y_train, batch_size, kmers, ids, cv = cv, shuffle = shuffle, training = True)
        fit_model_binary_sk(clf, train_generator, clf_file)
        train_generator.handle.close()
        y_pred_test = predict_binary_sk(clf_file, ids, test_generator)
        y_test = test_labels(test_generator)
        test_generator.handle.close()

    elif classifier in ["ridge","svm","mlr","kmeans","mnb"]:
        train_generator, test_generator = iter_generator(X_train, y_train, batch_size, kmers, ids, cv = cv, shuffle = shuffle, training = True)
        fit_model_multi_sk(clf, train_generator, clf_file)
        train_generator.handle.close()
        y_pred_test = predict_multi_sk(clf_file, ids, test_generator)
        y_test = test_labels(test_generator)
        test_generator.handle.close()

    elif classifier in ["attention","lstm","deeplstm"]:
        train_generator, val_generator, test_generator = iter_generator_keras(X_train, y_train, batch_size, kmers, ids, cv, classifier, shuffle = shuffle, training = True)
        fit_model_keras(clf, train_generator, val_generator, clf_file)
        train_generator.handle.close()
        val_generator.handle.close()
        y_pred_test = predict_binary_keras(clf_file, test_generator)
        y_test = test_labels(test_generator)
        test_generator.handle.close()

    elif classifier in ["lstm_attention","cnn","dbn","deepcnn"]:
        train_generator, val_generator, test_generator = iter_generator_keras(X_train, y_train, batch_size, kmers, ids, cv, classifier, shuffle = shuffle, training = True)
        fit_model_keras(clf, train_generator, val_generator, clf_file)
        train_generator.handle.close()
        val_generator.handle.close()
        y_pred_test = predict_multi_keras(clf_file, labels_list, test_generator)
        y_test = test_labels(test_generator)
        test_generator.handle.close()

    print(y_pred_test)
    print(y_test)

    training_cross_validation(y_pred_test, y_test, classifier)

    return clf

def fit_model(X_train, y_train, batch_size, kmers, ids, classifier, labels_list, clf, cv = 0, shuffle = True, verbose = True, clf_file = None):
    scaleX(X_train, y_train, batch_size, kmers, ids, verbose)

    if classifier in ["onesvm","linearsvm"]:
        generator = iter_generator(X_train, y_train, batch_size, kmers, cv = cv, shuffle = shuffle, training = True)
        fit_model_binary_sk(clf, generator, clf_file)
        generator.handle.close()
    elif classifier in ["ridge","svm","mlr","kmeans","mnb"]:
        generator = iter_generator(X_train, y_train, batch_size, kmers, cv = cv, shuffle = shuffle, training = True)
        fit_model_multi_sk(clf, generator, clf_file, labels_list)
        generator.handle.close()
    elif classifier in ["attention","lstm","deeplstm","lstm_attention","cnn","dbn","deepcnn"]:
        train_generator, val_generator = iter_generator_keras(X_train, y_train, batch_size, kmers, ids, cv, classifier, shuffle = shuffle, training = True)
        fit_model_keras(clf, train_generator, val_generator, clf_file)
        train_generator.handle.close()
        val_generator.handle.close()

def model_predict(clf_file, X, kmers_list, ids, classifier, nb_classes, labels_list, verbose = True):
    y = pd.Series(range(len(ids)))
    scaleX(X, y, 32, kmers_list, ids, verbose)

    if classifier in ["onesvm","linearsvm"]:
        generator = iter_generator(X, y, 1, kmers_list, ids, cv = 0, shuffle = False, training = False)
        predict = predict_binary_sk(clf_file, ids, generator)
        generator.handle.close()
    elif classifier in ["attention","lstm","deeplstm"]:
        generator = iter_generator_keras(X, y, 1, kmers_list, ids, 0, classifier, shuffle = False, training = False)
        predict = predict_binary_keras(clf_file, generator)
        generator.handle.close()
    elif classifier in ["ridge","svm","mlr","kmeans","mnb"]:
        generator = iter_generator(X, y, 1, kmers_list, ids, cv = 0, shuffle = False, training = False)
        predict = predict_multi_sk(clf_file, labels_list, generator)
        generator.handle.close()
    elif classifier in ["lstm_attention","cnn","dbn","deepcnn"]:
        generator = iter_generator_keras(X, y, 1, kmers_list, ids, 0, classifier, shuffle = False, training = False)
        predict = predict_multi_keras(clf_file, labels_list, generator)
        generator.handle.close()

    return predict

# Scikit-learn versions
################################################################################
def fit_model_binary_sk(clf, generator, clf_file):
    for i, (X, y) in enumerate(generator.iterator):
        clf.partial_fit(X, y, classes = np.array([-1,1]))
    dump(clf, clf_file)

def fit_model_multi_sk(clf, generator, clf_file, cls):
    for i, (X, y) in enumerate(generator.iterator):
        clf.partial_fit(X, y, classes = cls)
    dump(clf, clf_file)

def predict_binary_sk(clf_file, ids, generator):
    y_pred = np.empty(len(ids))

    clf = load(clf_file)
    for i, (X, y) in enumerate(generator.iterator):
        y_pred[i] = clf.predict(X)

    return y_pred

def predict_multi_sk(clf_file, labels_list, generator):
    threshold = 0.8
    y_pred = []

    clf = CalibratedClassifierCV(base_estimator = load(clf_file), cv = "prefit")
    for i, (X, y) in enumerate(generator.iterator):
        predict = clf.predict_proba(x)
        if np.argmax(predict[0]) >= threshold:
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
            epochs = 1,
            callbacks = [modelcheckpoint,early],
            use_multiprocessing = True,
            workers = os.cpu_count())

def predict_binary_keras(clf_file, generator):
    clf = load_model(clf_file)
    predict = clf.predict(generator,
                          use_multiprocessing = True,
                          workers = os.cpu_count())

    y_pred = np.round(predict.reshape(1, predict.size)[0])
    generator.handle.close()

    return y_pred

def predict_multi_keras(clf_file, labels_list, generator):
    threshold = 0.8
    y_pred = []

    clf = load_model(clf_file)
    predict = clf.predict(generator,
                          use_multiprocessing = True,
                          workers = os.cpu_count())
    for i in range(len(predict.reshape(1, predict.size))):
        if np.argmax(predict[i]) >= threshold:
            y_pred.append(labels_list[np.argmax(predict[i])])
        else:
            y_pred.append(-1)

    return y_pred
