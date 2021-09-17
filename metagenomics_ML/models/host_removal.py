import pandas as pd
import numpy as np

import sys
import os

from sklearn import metrics, model_selection
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

from data.build_data import load_Xy_data, save_Xy_data

from joblib import dump, load

__author__ = "nicolas"

def bacteria_extraction(metagenome_k_mers, database_k_mers, k, prefix, dataset, classifier = "svm", verbose = 1, saving_mode = "all"):
    bacterias_kmers_file = prefix + "_K{}_{}_Xy_bacterias_database_{}_data.hdf5.bz2".format(k, classifier, dataset)
    unclassified_kmers_file = prefix + "_K{}_{}_Xy_unclassified_database_{}_data.hdf5.bz2".format(k, classifier, dataset)
    clf_file = prefix + "_K{}_{}_bacteria_binary_classifier_{}_model.joblib".format(k, classifier, dataset)

    # Load if already exists
    if os.path.isfile(bacterias_kmers_file):
        bacterias = load_Xy_data(bacterias_kmers_file)

    else:
        # Get training dataset
        database_k_mers["X_train"] = pd.DataFrame(StandardScaler().fit_transform(database_k_mers["X_train"]), columns = database_k_mers["kmers_list"], index = database_k_mers["ids"])
        database_k_mers["y_train"][:] = 1
        database_k_mers["y_train"] = pd.DataFrame(database_k_mers["y_train"].astype(np.int32), index = database_k_mers["ids"])

        # Training data
        X_train = database_k_mers["X_train"]
        y_train = database_k_mers["y_train"]

        if os.path.isfile(clf_file):
            clf = load(clf_file)

        else:
            # Train/test classifier
            clf = training(X_train, y_train, classifier = classifier, verbose = verbose)

        # Classify sequences into bacterias / unclassified and return k-mers profiles
        bacterias, unclassified = extract_bacteria(clf, pd.DataFrame(metagenome_k_mers["X_train"], columns = metagenome_k_mers["kmers_list"], index = metagenome_k_mers["ids"]), verbose = verbose)

        if saving_mode != "none":
            save_data_clf(bacterias, unclassified, clf, bacterias_kmers_file, unclassified_kmers_file, clf_file, saving_mode)

    return bacterias

def extract_bacteria(clf, k_mers, verbose = 1):
    if verbose:
        print("Extracting predicted bacterias")
    predict = clf.predict(k_mers)
    bacterias = pd.DataFrame(columns = k_mers.columns)
    unclassified = pd.DataFrame(columns = k_mers.columns)
    for i in range(len(predict)):
        if predict[i] == 1:
            bacterias = bacterias.append(pd.DataFrame(k_mers.iloc[i]).transpose())
        elif predict[i] == -1:
            unclassified = unclassified.append(pd.DataFrame(k_mers.iloc[i]).transpose())

    return bacterias, unclassified

def training(X_train, y_train, classifier = "svm", verbose = 1):
    if classifier == "svm":
        if verbose:
            print("Training binary bacterial extractor with One Class SVM")
        clf = OneClassSVM(gamma = 'scale', kernel = "rbf", nu = 0.5)
    elif classifier == "lof":
        if verbose:
            print("Training binary bacterial extractor with Local Outlier Factor")
        clf = LocalOutlierFactor(contamination = 0.5, novelty = True, n_jobs = -1)
    else:
        print("Classifier type unknown !!! \n Models implemented at this moment are :  One Class SVM (svm) and Local Outlier Factor (lof)")
        sys.exit()

    clf.fit(X_train, y_train)

    return clf

# Save extracted k-mers profiles for bacterias and unclassified
# Depends on wich saving mode given by user
def save_data_clf(bacterias, unclassified, clf, bacterias_kmers_file, unclassified_kmers_file, clf_file, saving_mode):
    if saving_mode == "all":
        save_Xy_data(bacterias, bacterias_kmers_file)
        save_Xy_data(unclassified, unclassified_kmers_file)
        dump(clf, clf_file)
    elif saving_mode == "both":
        save_Xy_data(bacterias, bacterias_kmers_file)
        save_Xy_data(unclassified, unclassified_kmers_file)
    elif saving_mode == "bacterias_model":
        save_Xy_data(bacterias, bacterias_kmers_file)
        dump(clf, clf_file)
    elif saving_mode == "unclassified_model":
        save_Xy_data(unclassified, unclassified_kmers_file)
        dump(clf, clf_file)
    elif saving_mode == "bacterias":
        save_Xy_data(bacterias, bacterias_kmers_file)
    elif saving_mode == "unclassified":
        save_Xy_data(unclassified, unclassified_kmers_file)
    elif saving_mode == "model":
        dump(clf, clf_file)
