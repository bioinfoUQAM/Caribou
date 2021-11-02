
import pandas as pd
import numpy as np
import tables as tb

import os
import sys

from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from data.generators import DataGenerator, iter_generator_keras

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

from joblib import dump, load

import warnings

__author__ = "nicolas"

# Ignore warnings
warnings.filterwarnings("ignore")

# Load data from file
def load_Xy_data(Xy_file):
    if os.path.basename(Xy_file).split(sep = ".")[1] == "npz":
        with np.load(Xy_file, allow_pickle=True) as f:
            return f['data'].tolist()

# Save data to file
def save_Xy_data(data, Xy_file):
    if type(data) == pd.core.frame.DataFrame:
        with tb.open_file(Xy_file, "a") as handle:
            array = handle.create_carray("/", "data", obj = np.array(data,dtype=np.float32))
    elif type(data) == dict:
        np.savez(Xy_file, data=data)

# Model training cross-validation stats
def training_cross_validation(y_pred_test, y_test, classifier):
    print("Cross validating classifier : " + str(classifier))
    print("Confidence matrix : ")
    print(metrics.confusion_matrix(y_pred_test, y_test))
    print("Precision : {}".format(str(metrics.precision_score(y_pred_test, y_test))))
    print("Recall : {}".format(str(metrics.recall_score(y_pred_test, y_test))))
    print("F-score : {}".format(str(metrics.f1_score(y_pred_test, y_test))))

def fit_predict_cv(X_train, y_train, batch_size, kmers, ids, classifier, clf, cv = 1, shuffle = True, verbose = True, model_file = None):
    # Scale X_train dataset
    scaleX(X_train, y_train, batch_size, kmers, ids, verbose)

    if classifier in ["onesvm","linearsvm"]:
        X_test = pd.DataFrame()
        y_test = pd.DataFrame()
        generator = DataGenerator(X_train, y_train, batch_size, kmers, ids, cv = cv, shuffle = shuffle)
        for i, (X, y) in enumerate(generator.iterator_train):
            clf.partial_fit(X, y, classes = np.array([-1,1]))
            dump(clf, model_file)
        for i, (X, y) in enumerate(generator.iterator_test):
            try:
                if X_test.empty and y_test.empty:
                    X_test = pd.DataFrame(X)
                    y_test = pd.DataFrame(y)
                else:
                    X_test.append(X)
                    y_test.append(y)
            except:
                print("File too large to cross validate on RAM")
                sys.exit()
        generator.handle.close()
        clf = load(model_file)
        y_pred_test = clf.predict(X_test)

    elif classifier in ["attention","lstm","cnn","deeplstm"]:
        modelcheckpoint = ModelCheckpoint(filepath=model_file,monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
        early = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
        train_generator, val_generator, test_generator = iter_generator_keras(X_train, y_train, batch_size, kmers, ids, cv, classifier, shuffle = shuffle, training = True)
        clf.fit(train_generator,
                validation_data = val_generator,
                epochs = 100,
                callbacks = [modelcheckpoint,early],
                use_multiprocessing = True,
                workers = os.cpu_count())

        clf = load_model(model_file)
        predict = clf.predict(x = test_generator,
                           use_multiprocessing = True,
                           workers = os.cpu_count())

        y_pred_test = np.round(predict.copy())
        y_test = pd.DataFrame(np.zeros((len(y_pred_test),1), dtype = int))
        for i in range(len(y_pred_test)):
            if classifier not in ["lstm","deeplstm"]:
                y_test.iloc[i,0] = test_generator.labels[test_generator.positions_list[i]]
            else:
                y_test.iloc[i,0] = test_generator.labels[test_generator.positions_list[i]]
        train_generator.handle.close()
        val_generator.handle.close()
        test_generator.handle.close()
    return y_pred_test, y_test, clf

def fit_model(X_train, y_train, batch_size, kmers, ids, classifier, clf, cv = 1, shuffle = True, verbose = True, model_file = None):
    # Scale X_train dataset
    scaleX(X_train, y_train, batch_size, kmers, ids, verbose)

    if classifier in ["onesvm","linearsvm"]:
        generator = DataGenerator(X_train, y_train, batch_size, kmers, cv = cv, shuffle = shuffle)
        for i, (X, y) in enumerate(generator.iterator_train):
            clf.partial_fit(X, y, classes = np.array([-1,1]))
        generator.handle.close()
        dump(clf, model_file)
    elif classifier in ["attention","lstm","cnn","deeplstm"]:
        modelcheckpoint = ModelCheckpoint(filepath=model_file,monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
        early = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
        train_generator, val_generator = iter_generator_keras(X_train, y_train, batch_size, kmers, ids, cv, classifier, shuffle = shuffle)
        clf.fit(train_generator,
                validation_data = val_generator,
                epochs = 100,
                callbacks = [modelcheckpoint,early],
                use_multiprocessing = True,
                workers = os.cpu_count())

        train_generator.handle.close()
        val_generator.handle.close()
    return clf

def model_predict(model_file, X, kmers_list, ids, classifier, verbose = True):
    predict = np.empty(len(ids))
    y = pd.Series(range(len(ids)))
    # Scale dataset
    scaleX(X, y, 32, kmers_list, ids, verbose)

    if classifier in ["onesvm","linearsvm"]:
        generator = DataGenerator(X, y, 1, kmers_list, ids, cv = 0, shuffle = False)
        clf = load(model_file)
        for i, (X, y) in enumerate(generator.iterator):
            predict[i] = clf.predict(X)
        generator.handle.close()
    elif classifier in ["attention","lstm","cnn","deeplstm"]:
        generator = iter_generator_keras(X, y, 1, kmers_list, ids, 0, classifier, shuffle = False, training = False)
        clf = load_model(model_file)
        predict = clf.predict(x = generator,
                              use_multiprocessing = True,
                              workers = os.cpu_count())
        generator.handle.close()
        y_pred = np.round(predict.copy())
    return y_pred

def save_predicted_kmers(positions_list, y, kmers_list, ids, infile, outfile):
    data = False
    generator = DataGenerator(infile, y, 1, kmers_list, ids, cv = 0, shuffle = False)
    with tb.open_file(outfile, "a") as handle:
        for i, (X, y) in enumerate(generator.iterator):
            if i in positions_list and not data:
                data = handle.create_earray("/", "data", obj = np.array(X))
            elif i in positions_list and data:
                data.append(np.array(X))
    generator.handle.close()


def scaleX(X_train, y_train, batch_size, kmers, ids, cv = 0, shuffle = False, verbose = True):
    try:
        generator = DataGenerator(X_train, y_train, batch_size, kmers, ids, cv = 0, shuffle = False)
        for i, (X, y) in enumerate(generator.iterator):
            scaler = StandardScaler().partial_fit(X)
        generator.handle.close()
        with tb.open_file(X_train, "a") as handle:
            handle.root.scaled = handle.create_carray("/", "scaled", obj = np.array(scaler.transform(handle.root.data.read()),dtype=np.float32))
    except tb.exceptions.NodeError:
        if verbose:
            print("Data already scaled")

def merge_database_host(database_data, host_data):
    merged_data = dict()

    path, ext = os.path.splitext(database_data["X"])
    merged_file = "{}_host_merged{}".format(path, ext)

    merged_data["X"] = merged_file
    merged_data["y"] = np.concatenate((database_data["y"], host_data["y"]))
    merged_data["ids"] = database_data["ids"] + host_data["ids"]
    merged_data["kmers_list"] = list(set(database_data["kmers_list"]).union(host_data["kmers_list"]))
    merged_data["taxas"] = max(database_data["taxas"], host_data["taxas"], key = len)

    generator_database = DataGenerator(database_data["X"], database_data["y"], 32, database_data["kmers_list"], database_data["ids"], cv = 0, shuffle = False)
    generator_host = DataGenerator(host_data["X"], host_data["y"], 32, host_data["kmers_list"], host_data["ids"], cv = 0, shuffle = False)
    if not os.path.isfile(merged_file):
        data = False
        with tb.open_file(merged_file, "a") as handle:
            for (X_d, y_d), (X_h, y_h) in zip(generator_database.iterator, generator_host.iterator):
                if not data:
                    data = handle.create_earray("/", "data", obj = np.array(pd.merge(X_d, X_h, how = "outer")))
                else:
                    data.append(np.array(pd.merge(X_d, X_h, how = "outer")))
        generator_database.handle.close()
        generator_host.handle.close()

    return merged_data
