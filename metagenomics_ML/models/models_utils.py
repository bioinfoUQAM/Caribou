
import pandas as pd
import numpy as np
import tables as tb

import os
import sys

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
#from sklearn.kernel_approximation import Nystroem
#from sklearn.linear_model import SGDOneClassSVM

from data.generators import DataGenerator, iter_generator_keras

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

from joblib import dump, load

import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Utils for all types of models
################################################################################

# X data scaling
def scaleX(X_data, y_data, batch_size, kmers, ids, cv = 0, shuffle = False, verbose = True):
    try:
        scaler = StandardScaler()
        generator = DataGenerator(X_data, y_data, batch_size, kmers, ids, cv = 0, shuffle = False)
        for i, (X, y) in enumerate(generator.iterator):
            scaler.partial_fit(X)
        generator.handle.close()
        with tb.open_file(X_data, "a") as handle:
            handle.root.scaled = handle.create_carray("/", "scaled", obj = np.array(scaler.transform(handle.root.data.read()),dtype=np.float32))
    except tb.exceptions.NodeError:
        if verbose:
            print("Data already scaled")

"""
# To exclude before classification
def exclude_unclassified(X_db, y_db, X_data, batch_size, kmers, ids, unclassified_kmers_file, verbose):
    predict = np.empty(len(ids))
    y_data = pd.Series(range(len(ids)))
    unclassified = []

    transform = Nystroem(n_jobs = -1)
    clf = SGDOneClassSVM(nu = 0.05, tol = 1e-4)

    generator_db = DataGenerator(X_db, y_db, batch_size, kmers, ids, cv = 0, shuffle = True)
    generator_data = DataGenerator(X_data, y_data, 1, kmers, ids, cv = 0, shuffle = False)

    for i, (X, y) in enumerate(generator_db.iterator):
        X = transform.fit_transform(X)
        clf.partial_fit(X, y, classes = np.array([-1,1]))
    for i, (X, y) in enumerate(generator_data.iterator):
        predict[i] = clf.predict(X)

    generator_db.handle.close()
    generator_data.handle.close()

    for i in range(len(predict)):
        if predict[i] == -1:
            unclassified.append(i)
    save_predicted_kmers(bacteria, pd.Series(range(len(ids))), kmers_list, ids, X, bacteria_kmers_file)
    save_predicted_kmers(unclassified, pd.Series(range(len(ids))), kmers_list, ids, X, unclassified_kmers_file)
"""

# Model training cross-validation
def training_cross_validation(y_pred_test, y_test, classifier):
    print("Cross validating classifier : " + str(classifier))
    print("Confidence matrix : ")
    print(metrics.confusion_matrix(y_pred_test, y_test))
    print("Precision : {}".format(str(metrics.precision_score(y_pred_test, y_test))))
    print("Recall : {}".format(str(metrics.recall_score(y_pred_test, y_test))))
    print("F-score : {}".format(str(metrics.f1_score(y_pred_test, y_test))))

# Fitting and predicting with models
################################################################################

def fit_predict_cv(X_train, y_train, batch_size, kmers, ids, classifier, clf, cv = 1, shuffle = True, verbose = True, clf_file = None):
    scaleX(X_train, y_train, batch_size, kmers, ids, verbose)

    if classifier in ["onesvm","linearsvm","ridge","svm","mlr","kmeans","mnb"]:
        X_test = pd.DataFrame()
        y_test = pd.DataFrame()
        generator = DataGenerator(X_train, y_train, batch_size, kmers, ids, cv = cv, shuffle = shuffle)
        for i, (X, y) in enumerate(generator.iterator_train):
            clf.partial_fit(X, y, classes = np.array([-1,1]))
            dump(clf, clf_file)
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
        clf = load(clf_file)
        y_pred_test = clf.predict(X_test)

    elif classifier in ["attention","lstm","deeplstm","lstm_attention","cnn","dbn","deepcnn"]:
        modelcheckpoint = ModelCheckpoint(filepath=clf_file,monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
        early = EarlyStopping(monitor='val_accuracy', mode='min', verbose=1, patience=10)
        train_generator, val_generator, test_generator = iter_generator_keras(X_train, y_train, batch_size, kmers, ids, cv, classifier, shuffle = shuffle, training = True)
        clf.fit(x = train_generator,
                validation_data = val_generator,
                epochs = 1,
                callbacks = [modelcheckpoint,early],
                use_multiprocessing = True,
                workers = os.cpu_count())

        clf = load_model(clf_file)
        predict = clf.predict(test_generator,
                              use_multiprocessing = True,
                              workers = os.cpu_count())

        print(predict)
        y_pred_test = np.round(predict.copy())
        print(y_pred_test)
        y_test = pd.DataFrame(np.zeros((len(y_pred_test),1), dtype = int))
        for i in range(len(y_pred_test)):
            if classifier not in ["lstm","deeplstm","lstm_attention","cnn","dbn","deepcnn"]:
                y_test.iloc[i,0] = test_generator.labels[test_generator.positions_list[i]]
            else:
                y_test.iloc[i,0] = test_generator.labels[test_generator.positions_list[i]]
        train_generator.handle.close()
        val_generator.handle.close()
        test_generator.handle.close()

    training_cross_validation(y_pred_test, list(y_test[0]), classifier)

    return clf

def fit_model(X_train, y_train, batch_size, kmers, ids, classifier, clf, cv = 1, shuffle = True, verbose = True, clf_file = None):
    scaleX(X_train, y_train, batch_size, kmers, ids, verbose)

    if classifier in ["onesvm","linearsvm","ridge","svm","mlr","kmeans","mnb"]:
        generator = DataGenerator(X_train, y_train, batch_size, kmers, cv = cv, shuffle = shuffle)
        for i, (X, y) in enumerate(generator.iterator_train):
            clf.partial_fit(X, y, classes = np.array([-1,1]))
        generator.handle.close()
        dump(clf, clf_file)
    elif classifier in ["attention","lstm","deeplstm","lstm_attention","cnn","dbn","deepcnn"]:
        modelcheckpoint = ModelCheckpoint(filepath=clf_file,monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
        early = EarlyStopping(monitor='val_accuracy', mode='min', verbose=1, patience=10)
        train_generator, val_generator = iter_generator_keras(X_train, y_train, batch_size, kmers, ids, cv, classifier, shuffle = shuffle)
        clf.fit(x = train_generator,
                validation_data = val_generator,
                epochs = 100,
                callbacks = [modelcheckpoint,early],
                use_multiprocessing = True,
                workers = os.cpu_count())

        train_generator.handle.close()
        val_generator.handle.close()
    return clf

def model_predict(clf_file, X, kmers_list, ids, classifier, verbose = True):
    predict = np.empty(len(ids))
    prob = np.empty(len(ids))
    y = pd.Series(range(len(ids)))

    scaleX(X, y, 32, kmers_list, ids, verbose)

# OUTPUT : PROBABILITY OF EACH PREDICTED CLASSIFICATION
# SET THRESHOLD FOR EFFECTIVELY CLASSIFYING, ELSE CLASSIFY UNKNOWN
# When calling classifiers : probability = True -> clf.prefict_proba
# sklearn : clf.predict_proba, clf.scores, clf.decision_function
    if classifier in ["onesvm","linearsvm","ridge","svm","mlr","kmeans","mnb"]:
        generator = DataGenerator(X, y, 1, kmers_list, ids, cv = 0, shuffle = False)
        clf = load(clf_file)
        for i, (X, y) in enumerate(generator.iterator):
            predict[i] = clf.predict(X)
            prob[i] = clf.predict_proba(X)
        generator.handle.close()
        print(prob)
    elif classifier in ["attention","lstm","deeplstm","lstm_attention","cnn","dbn","deepcnn"]:
        generator = iter_generator_keras(X, y, 1, kmers_list, ids, 0, classifier, shuffle = False, training = False)
        clf = load_model(clf_file)
        predict = clf.predict(x = generator,
                              use_multiprocessing = True,
                              workers = os.cpu_count())
        generator.handle.close()
        y_pred = np.round(predict.copy())
    return y_pred

def predict_binary():
    print("To do")

def predict_multiclass():
    print("To do")
