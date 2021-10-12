
import pandas as pd
import numpy as np
import tables as tb

import os

from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from data.generators import DataGenerator

__author__ = "nicolas"

# Load data from file
def load_Xy_data(Xy_file):
    if os.path.basename(Xy_file).split(sep = ".")[1] == "npz":
        with np.load(Xy_file, allow_pickle=True) as f:
            return f['data'].tolist()

# Save data to file
def save_Xy_data(data,Xy_file):
    if type(data) == pd.core.frame.DataFrame:
        data.to_hdf(Xy_file, key='df', mode='w', complevel = 9, complib = 'bzip2')
    elif type(data) == dict:
        np.savez(Xy_file, data=data)

# Model training cross-validation stats
def training_cross_validation(y_pred_test, y_test, classifier):
    print("Cross validating classifier : " + str(classifier))

    print("y_pred_test : ")
    print(y_pred_test)
    print("y_test : ")
    print(y_test)

    print("Confidence matrix : ")
    print(str(metrics.confusion_matrix(y_pred_test, y_test)))
    print("Precision : " + str(metrics.precision_score(y_pred_test, y_test)))
    print("Recall : " + str(metrics.recall_score(y_pred_test, y_test)))
    print("F-score : " + str(metrics.f1_score(y_pred_test, y_test)))
    print("AUC ROC : " + str(metrics.roc_auc_score(y_pred_test, y_test)))

def fit_predict_cv(X_train, y_train, batch_size, kmers, classifier, clf, nb_classes, cv = 1, shuffle = True):
    if classifier == "lstm":

        train_generator, val_generator, test_generator = iter_generator_keras(X_train, y_train, batch_size, kmers, nb_classes = nb_classes, cv = cv, shuffle = shuffle)
        clf.fit_generator(generator = train_generator,
                            validation_data = val_generator,
                            use_multiprocessing = True,
                            workers = os.cpu_count())

        y_pred_test = clf.predict_generator(generator = test_generator,
                                            use_multiprocessing = True,
                                            workers = os.cpu_count())
    else:
        X_test = pd.DataFrame()
        y_test = pd.DataFrame()
        generator = DataGenerator(X_train, y_train, batch_size, kmers, nb_classes = nb_classes, cv = cv, shuffle = shuffle)
        for i, (X, y) in enumerate(generator.iterator_train):
                clf.partial_fit(X, y)
        for i, (X, y) in enumerate(generator.iterator_test):
            try:
                if X_test.empty and y_test.empty:
                    X_test = pd.DataFrame(X)
                    y_test = pd.DataFrame(y)
                else:
                    X_test.append(X)
                    y_test.append(y)
            except Error as e:
                print("File too large to cross validate on RAM")
                sys.exit()

        y_pred_test = clf.predict(X_test)
    return y_pred_test, clf

def fit_model(X_train, y_train, batch_size, kmers, classifier, clf, nb_classes, cv = 1, shuffle = True):
    if classifier == "lstm":
        train_generator, val_generator = iter_generator_keras(X_train, y_train, batch_size, kmers, nb_classes = nb_classes, cv = cv, shuffle = shuffle)
        clf.fit_generator(generator = train_generator,
                            validation_data = val_generator,
                            use_multiprocessing = True,
                            workers = os.cpu_count())
        clf.fit(X, y, epochs = 100, batch_size = 32)
    else:
        generator = DataGenerator(X_train, y_train, batch_size, kmers, nb_classes = nb_classes, cv = cv, shuffle = shuffle)
        for i, (X, y) in enumerate(generator.iterator):
            clf.partial_fit(X, y)

    return clf

def scaleX(X_train, y_train, batch_size, kmers, nb_classes, cv = 0, shuffle = False):
    generator = DataGenerator(X_train, y_train, batch_size, kmers, nb_classes, cv = cv, shuffle = shuffle)
    for i, (X, y) in enumerate(generator.iterator):
        StandardScaler().partial_fit(X)
    with tb.open_file(X_train, "w") as handle:
        handle.root.data = handle.create_carray("/", "data", obj = np.array(StandardScaler().transform(handle.root.data.read()),dtype=np.int32))
