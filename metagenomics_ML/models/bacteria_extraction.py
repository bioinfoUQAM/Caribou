
import pandas as pd

import sys
import os

from sklearn.linear_model import SGDOneClassSVM, SGDClassifier

from utils import *
from models.build_neural_networks import *

from joblib import dump, load

__author__ = "nicolas"

# TRIES TO TRAIN TWICE...
# TESTER GENERATORS
# Voir si peut utiliser celui de Keras avec sklearn classifiers

def bacteria_extraction(metagenome_k_mers, database_k_mers, k, prefix, dataset, classifier = "onesvm", batch_size = 32, verbose = 1, cv = 1, saving_host = 1, saving_unclassified = 1):
    bacteria_kmers_file = "{}_K{}_{}_Xy_bacteria_database_{}_data.h5f".format(prefix, k, classifier, dataset)
    host_kmers_file = prefix + "_K{}_{}_Xy_host_database_{}_data.h5f".format(k, classifier, dataset)
    unclassified_kmers_file = prefix + "_K{}_{}_Xy_unclassified_database_{}_data.h5f".format(k, classifier, dataset)
    clf_file ="{}_K{}_{}_bacteria_binary_classifier_{}_model.joblib".format(prefix, k, classifier, dataset)

    if verbose:
        print("Extracting bacteria sequences from data")

    # Load extracted data if already exists or train and extract bacteria depending on chosen method
    if os.path.isfile(bacteria_kmers_file):
        bacteria = load_Xy_data(bacteria_kmers_file)
    else:
        # Get training dataset and assign to variables
        if classifier == "onesvm" and isinstance(database_k_mers, tuple):
            print("Classifier One Class SVM cannot be used with host data!\nEither remove host data from config file or choose another bacteria extraction method.")
            sys.exit()
        elif classifier == "onesvm" and not isinstance(database_k_mers, tuple):
            X_train = database_k_mers["X"]
            y_train = pd.DataFrame(database_k_mers["y"], index = database_k_mers["ids"], columns = database_k_mers["taxas"]).loc[:,"domain"]
            y_train = y_train.replace("bacteria", 1)
        elif classifier != "onesvm" and isinstance(database_k_mers, tuple):
            database_k_mers = merge_database_host(database_k_mers[0], database_k_mers[1])
            X_train = database_k_mers["X"]
            y_train = pd.DataFrame(database_k_mers["y"], index = database_k_mers["ids"], columns = database_k_mers["taxas"]).loc[:,"domain"]
            y_train = y_train.replace("bacteria", 1)
            y_train = y_train.replace("host", -1)
        else:
            print("Only classifier One Class SVM can be used without host data!\nEither add host data in config file or choose classifier One Class SVM.")
            sys.exit()

        # If classifier exists load it or train if not
        if os.path.isfile(clf_file):
            clf = load(clf_file)
        else:
            clf = training(X_train, y_train, database_k_mers["kmers_list"], database_k_mers["ids"], classifier = classifier, batch_size = batch_size, verbose = verbose, cv = cv)
            #dump(clf, clf_file)

        # Classify sequences into bacteria / unclassified / host and build k-mers profiles for bacteria
        bacteria = extract_bacteria_sequences(clf, metagenome_k_mers["X"], metagenome_k_mers["kmers_list"], metagenome_k_mers["ids"], classifier, bacteria_kmers_file, host_kmers_file, unclassified_kmers_file, verbose = verbose, saving_host = saving_host, saving_unclassified = saving_unclassified)

    return bacteria

def extract_bacteria_sequences(clf, X, kmers_list, ids, classifier, bacteria_kmers_file, host_kmers_file, unclassified_kmers_file, verbose = 1, saving_host = 1, saving_unclassified = 1):

    predict = model_predict(clf, X, kmers_list, ids, classifier, verbose)
    bacteria = []
    host = []
    unclassified = []

    if classifier == "onesvm" and saving_unclassified:
        if verbose:
            print("Extracting predicted bacteria and unclassified sequences")
        for i in range(len(predict)):
            if predict[i] == 1:
                bacteria.append(i)
            elif predict[i] == -1:
                unclassified.append(i)
        save_predicted_kmers(bacteria, pd.Series(range(len(ids))), kmers_list, ids, X, bacteria_kmers_file)
        save_predicted_kmers(unclassified, pd.Series(range(len(ids))), kmers_list, ids, X, unclassified_kmers_file)
    elif classifier != "onesvm" and saving_host and saving_unclassified:
        if verbose:
            print("Extracting predicted bacteria, host and unclassified sequences")
        for i in range(len(predict)):
            if predict[i] == 1:
                bacteria.append(i)
            elif predict[i] == -1:
                host.append(i)
            elif (-1 < predict[i] < 1):
                unclassified.append()
        save_predicted_kmers(bacteria, pd.Series(range(len(ids))), kmers_list, ids, X, bacteria_kmers_file)
        save_predicted_kmers(host, pd.Series(range(len(ids))), kmers_list, ids, X, host_kmers_file)
        save_predicted_kmers(unclassified, pd.Series(range(len(ids))), kmers_list, ids, X, unclassified_kmers_file)
    elif classifier != "onesvm" and saving_host:
        if verbose:
            print("Extracting predicted bacteria and host sequences")
        for i in range(len(predict)):
            if predict[i] == 1:
                bacteria.append(i)
            elif predict[i] == -1:
                host.append(i)
        save_predicted_kmers(bacteria, pd.Series(range(len(ids))), kmers_list, ids, X, bacteria_kmers_file)
        save_predicted_kmers(host, pd.Series(range(len(ids))), kmers_list, ids, X, host_kmers_file)
    else:
        if verbose:
            print("Extracting predicted bacteria sequences")
        for i in range(len(predict)):
            if predict[i] == 1:
                bacteria.append(i)
        save_predicted_kmers(bacteria, pd.Series(range(len(ids))), kmers_list, ids, X, bacteria_kmers_file)

    bacteria = {}
    bacteria["X"] = str(bacteria_kmers_file)
    bacteria["kmers_list"] = kmers_list
    bacteria["ids"] = ids

    return bacteria

def training(X_train, y_train, kmers, ids, classifier = "onesvm", batch_size = 32, verbose = 1, cv = 1):
    if classifier == "onesvm":
        if verbose:
            print("Training bacterial extractor with One Class SVM")
        clf = SGDOneClassSVM(nu = 0.05, tol = 1e-4)
    elif classifier == "linearsvm":
        if verbose:
            print("Training bacterial / host classifier with Linear SVM")
        clf = SGDClassifier(early_stopping = True, n_jobs = -1)
    elif classifier == "virnet":
        if verbose:
            print("Training bacterial / host classifier based on VirNet method")
        clf = build_virnet()
    elif classifier == "seeker":
        if verbose:
            print("Training bacterial / host classifier based on Seeker LSTM method")
        clf = build_seeker()
    else:
        print("Classifier type unknown !!! \n Models implemented at this moment are \n bacteria isolator :  One Class SVM (onesvm)\n bacteria/host classifiers : Linear SVM (multiSVM), Random forest (forest), KNN clustering (knn) and LSTM RNN (lstm)")
        sys.exit()

    """
    # Maybe implement if have time / useful
    elif classifier == "gradl":
        if verbose:
            print("Training bacterial extractor based on GRaDL method")
            clf = build_gradl()
    """

    if cv:
        y_pred_test, y_test, clf = fit_predict_cv(X_train, y_train, batch_size, kmers, ids, classifier, clf, cv = cv, shuffle = True, verbose = verbose)
        training_cross_validation(y_pred_test, list(y_test[0]), classifier)
    else:
        clf = fit_model(X_train, y_train, batch_size, kmers, ids, classifier, clf, cv = cv, shuffle = True, verbose = verbose)

    return clf
