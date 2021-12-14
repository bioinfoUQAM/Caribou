
import pandas as pd

import sys
import os

from sklearn.linear_model import SGDOneClassSVM, SGDClassifier

from keras.models import load_model

from Caribou.utils import *
from Caribou.models.models_utils import *
from Caribou.models.build_neural_networks import *

from joblib import load

__author__ = "Nicolas de Montigny"

__all__ = ['bacteria_extraction','training','extract_bacteria_sequences']

def bacteria_extraction(metagenome_k_mers, database_k_mers, k, outdirs, dataset, classifier = "deeplstm", batch_size = 32, verbose = 1, cv = 1, saving_host = 1, saving_unclassified = 1, n_jobs = 1):
    train = False

    bacteria_data_file = "{}_K{}_{}_Xy_bacteria_database_{}_data.npz".format(outdirs["data_dir"], k, classifier, dataset)
    bacteria_kmers_file = "{}_K{}_{}_Xy_bacteria_database_{}_data.h5f".format(outdirs["data_dir"], k, classifier, dataset)
    host_kmers_file ="{}_K{}_{}_Xy_host_database_{}_data.h5f".format(outdirs["data_dir"], k, classifier, dataset)
    unclassified_kmers_file ="{}_K{}_{}_Xy_unclassified_database_{}_data.h5f".format(outdirs["data_dir"], k, classifier, dataset)

    if classifier in ["onesvm","linearsvm"]:
        clf_file ="{}_K{}_{}_bacteria_binary_classifier_{}_model.jb".format(outdirs["models_dir"], k, classifier, dataset)
        if not os.path.isfile(clf_file):
            train = True
    elif classifier in ["attention","lstm","deeplstm"]:
        clf_file ="{}_K{}_{}_bacteria_binary_classifier_{}_model".format(outdirs["models_dir"], k, classifier, dataset)
        if not os.path.isdir(clf_file):
            train = True

    if verbose:
        print("Extracting bacteria sequences from data")

    # Load extracted data if already exists or train and extract bacteria depending on chosen method
    if os.path.isfile(bacteria_data_file):
        bacteria = load_Xy_data(bacteria_data_file)
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
        if train == True:
            clf_file = training(X_train, y_train, database_k_mers["kmers_list"], k, database_k_mers["ids"], [-1, 1], outdirs["plots_dir"] if cv else None, classifier = classifier, batch_size = batch_size, verbose = verbose, cv = cv, clf_file = clf_file, n_jobs = n_jobs)
        # Classify sequences into bacteria / unclassified / host and build k-mers profiles for bacteria
        bacteria = extract_bacteria_sequences(clf_file, metagenome_k_mers["X"], metagenome_k_mers["kmers_list"], metagenome_k_mers["ids"], classifier, [-1, 1], bacteria_kmers_file, host_kmers_file, unclassified_kmers_file, verbose = verbose, saving_host = saving_host, saving_unclassified = saving_unclassified)
        save_Xy_data(bacteria, bacteria_data_file)

    return bacteria

def training(X_train, y_train, kmers, k, ids, labels_list, outdir_plots, classifier = "deeplstm", batch_size = 32, verbose = 1, cv = 1, clf_file = None, n_jobs = 1):
    if classifier == "onesvm":
        if verbose:
            print("Training bacterial extractor with One Class SVM")
        clf = SGDOneClassSVM(nu = 0.05, tol = 1e-4)
    elif classifier == "linearsvm":
        if verbose:
            print("Training bacterial / host classifier with Linear SVM")
        clf = SGDClassifier(early_stopping = False, n_jobs = -1)
    elif classifier == "attention":
        if verbose:
            print("Training bacterial / host classifier based on Attention Weighted Neural Network")
        clf = build_attention(len(kmers))
    elif classifier == "lstm":
        if verbose:
            print("Training bacterial / host classifier based on LSTM Neural Network")
        clf = build_LSTM(len(kmers), batch_size)
    elif classifier == "deeplstm":
        if verbose:
            print("Training bacterial / host classifier based on Deep LSTM Neural Network")
        clf = build_deepLSTM(len(kmers), batch_size)
    else:
        print("Bacteria extractor unknown !!!\n\tModels implemented at this moment are :\n\tBacteria isolator :  One Class SVM (onesvm)\n\tBacteria/host classifiers : Linear SVM (linearsvm)\n\tNeural networks : Attention (attention), LSTM (lstm) and Deep LSTM (deeplstm)")
        sys.exit()

    if cv:
        clf_file = cross_validation_training(X_train, y_train, batch_size, kmers, ids, classifier, labels_list, outdir_plots, clf, cv = cv, verbose = verbose, clf_file = clf_file, n_jobs = n_jobs)
    else:
        fit_model(X_train, y_train, batch_size, kmers, ids, classifier, labels_list, clf, cv = cv, shuffle = True, verbose = verbose, clf_file = clf_file)

    return clf_file

def extract_bacteria_sequences(clf_file, X, kmers_list, ids, classifier, labels_list, bacteria_kmers_file, host_kmers_file, unclassified_kmers_file, verbose = 1, saving_host = 1, saving_unclassified = 1):

    predict = model_predict(clf_file, X, kmers_list, ids, classifier, nb_classes = 2, labels_list = labels_list, verbose = verbose)
    bacteria = []
    host = []
    unclassified = []

    # OneSVM sklearn
    if classifier == "onesvm" and saving_unclassified:
        if verbose:
            print("Extracting predicted bacteria and unclassified sequences")
        for i in range(len(predict)):
            if predict[i] == 1:
                bacteria.append(i)
            elif predict[i] == -1:
                unclassified.append(i)
        save_predicted_kmers(bacteria, pd.Series(range(len(ids))), kmers_list, ids, X, bacteria_kmers_file, "binary")
        save_predicted_kmers(unclassified, pd.Series(range(len(ids))), kmers_list, ids, X, unclassified_kmers_file, "binary")
    # Keras classifiers
    elif classifier not in ["onesvm","linearsvm"] and saving_host and saving_unclassified:
        if verbose:
            print("Extracting predicted bacteria, host and unclassified sequences")
        for i in range(len(predict)):
            if predict[i] == 1:
                bacteria.append(i)
            elif predict[i] == 0:
                unclassified.append(i)
            elif predict[i] == -1:
                host.append(i)
        save_predicted_kmers(bacteria, pd.Series(range(len(ids))), kmers_list, ids, X, bacteria_kmers_file, "binary")
        save_predicted_kmers(host, pd.Series(range(len(ids))), kmers_list, ids, X, host_kmers_file, "binary")
        save_predicted_kmers(unclassified, pd.Series(range(len(ids))), kmers_list, ids, X, unclassified_kmers_file, "binary")
    # LinearSVM sklearn
    elif classifier != "onesvm" and saving_host:
        if verbose:
            print("Extracting predicted bacteria and host sequences")
        for i in range(len(predict)):
            if predict[i] == 1:
                bacteria.append(i)
            elif predict[i] == -1:
                host.append(i)
        save_predicted_kmers(bacteria, pd.Series(range(len(ids))), kmers_list, ids, X, bacteria_kmers_file, "binary")
        save_predicted_kmers(host, pd.Series(range(len(ids))), kmers_list, ids, X, host_kmers_file, "binary")
    # Only saving bacterias
    else:
        if verbose:
            print("Extracting predicted bacteria sequences")
        for i in range(len(predict)):
            if predict[i] == 1:
                bacteria.append(i)
        save_predicted_kmers(bacteria, pd.Series(range(len(ids))), kmers_list, ids, X, bacteria_kmers_file, "binary")

    bacteria_data = {}
    bacteria_data["X"] = str(bacteria_kmers_file)
    bacteria_data["kmers_list"] = kmers_list
    bacteria_data["ids"] = [ids[i] for i in bacteria]

    return bacteria_data
