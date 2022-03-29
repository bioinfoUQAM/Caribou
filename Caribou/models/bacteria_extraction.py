
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

def bacteria_extraction(metagenome_k_mers, database_k_mers, k, outdirs, dataset, training_epochs, classifier = "deeplstm", batch_size = 32, verbose = 1, cv = 1, n_jobs = 1):
    # classified_data is a dictionnary containing data dictionnaries at each classified level:
    # {taxa:{"X":string to Xy_data file.hdf5,"kmers_list":list of kmers,"ids":list of ids which where classified at that taxa}}
    classified_data = {"order":["bacteria","host","unclassified"]}

    train = False

    bacteria_data_file = "{}Xy_bacteria_database_K{}_{}_{}_data.npz".format(outdirs["data_dir"], k, classifier, dataset)
    bacteria_kmers_file = "{}Xy_bacteria_database_K{}_{}_{}_data.h5f".format(outdirs["data_dir"], k, classifier, dataset)
    host_kmers_file = "{}Xy_host_database_K{}_{}_{}_data.h5f".format(outdirs["data_dir"], k, classifier, dataset)
    host_data_file = "{}Xy_host_database_K{}_{}_{}_data.npz".format(outdirs["data_dir"], k, classifier, dataset)
    unclassified_kmers_file = "{}Xy_unclassified_database_K{}_{}_{}_data.h5f".format(outdirs["data_dir"], k, classifier, dataset)
    unclassified_data_file = "{}Xy_unclassified_database_K{}_{}_{}_data.npz".format(outdirs["data_dir"], k, classifier, dataset)

    if classifier in ["onesvm","linearsvm"]:
        clf_file ="{}bacteria_binary_classifier_K{}_{}_{}_model.jb".format(outdirs["models_dir"], k, classifier, dataset)
        if not os.path.isfile(clf_file):
            train = True
    elif classifier in ["attention","lstm","deeplstm"]:
        clf_file ="{}bacteria_binary_classifier_K{}_{}_{}_model".format(outdirs["models_dir"], k, classifier, dataset)
        if not os.path.isdir(clf_file):
            train = True

    # Load extracted data if already exists or train and extract bacteria depending on chosen method
    if os.path.isfile(bacteria_data_file):
        classified_data["bacteria"] = load_Xy_data(bacteria_data_file)
        try:
            classified_data["host"] = load_Xy_data(host_data_file)
        except:
            pass
        classified_data["unclassified"] = load_Xy_data(unclassified_data_file)
        if verbose:
            print("Bacteria sequences already extracted. Skipping this step")
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
            y_train = pd.DataFrame(database_k_mers["y"], index = database_k_mers["ids"], columns = database_k_mers["taxas"]).loc[:,"domain"].str.lower()
            y_train = y_train.replace("bacteria", 1)
            y_train = y_train.replace("host", -1)
        else:
            print("Only classifier One Class SVM can be used without host data!\nEither add host data in config file or choose classifier One Class SVM.")
            sys.exit()

        # If classifier exists load it or train if not
        if train is True:
            clf_file = training(X_train, y_train, database_k_mers["kmers_list"], k, database_k_mers["ids"], [-1, 1], outdirs["plots_dir"] if cv else None, training_epochs, classifier = classifier, batch_size = batch_size, verbose = verbose, cv = cv, clf_file = clf_file, n_jobs = n_jobs)
        # Classify sequences into bacteria / unclassified / host and build k-mers profiles for bacteria
        if metagenome_k_mers is not None:
            classified_data = extract_bacteria_sequences(clf_file, classified_data, metagenome_k_mers["X"], metagenome_k_mers["kmers_list"], metagenome_k_mers["ids"], classifier, [-1, 1], bacteria_kmers_file, host_kmers_file, unclassified_kmers_file, verbose = verbose)
            save_Xy_data(classified_data["bacteria"], bacteria_data_file)
            save_Xy_data(classified_data["unclassified"], unclassified_data_file)
            if classifier != 'onesvm':
                save_Xy_data(classified_data["host"], host_data_file)
            return classified_data


def training(X_train, y_train, kmers, k, ids, labels_list, outdir_plots, training_epochs, classifier = "deeplstm", batch_size = 32, verbose = 1, cv = 1, clf_file = None, n_jobs = 1):
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
            print("Training bacterial / host classifier based on Shallow LSTM Neural Network")
        clf = build_LSTM(len(kmers), batch_size)
    elif classifier == "deeplstm":
        if verbose:
            print("Training bacterial / host classifier based on Deep LSTM Neural Network")
        clf = build_deepLSTM(len(kmers), batch_size)
    else:
        print("Bacteria extractor unknown !!!\n\tModels implemented at this moment are :\n\tBacteria isolator :  One Class SVM (onesvm)\n\tBacteria/host classifiers : Linear SVM (linearsvm)\n\tNeural networks : Attention (attention), Shallow LSTM (lstm) and Deep LSTM (deeplstm)")
        sys.exit()

    if cv:
        clf_file = cross_validation_training(X_train, y_train, batch_size, kmers, ids, classifier, labels_list, outdir_plots, clf, training_epochs, cv = cv, verbose = verbose, clf_file = clf_file, n_jobs = n_jobs)
    else:
        fit_model(X_train, y_train, batch_size, kmers, ids, classifier, labels_list, clf, training_epochs, cv = cv, shuffle = True, verbose = verbose, clf_file = clf_file)

    return clf_file

def extract_bacteria_sequences(clf_file, classified_data, X, kmers_list, ids, classifier, labels_list, bacteria_kmers_file, host_kmers_file, unclassified_kmers_file = None, verbose = 1):

    predict = model_predict(clf_file, X, kmers_list, ids, classifier, nb_classes = 2, labels_list = labels_list, verbose = verbose)
    bacteria = []
    host = []
    unclassified = []

    if verbose:
        print("Extracting predicted bacteria sequences")

    # Extract positions for each classification possible
    if classifier == "onesvm":
        for i in range(len(predict)):
            if predict[i] == 1:
                bacteria.append(i)
            elif predict[i] == -1:
                unclassified.append(i)
    else:
        for i in range(len(predict)):
            if predict[i] == 1:
                bacteria.append(i)
            elif predict[i] == 0:
                unclassified.append(i)
            elif predict[i] == -1:
                host.append(i)

    # Save data
    if classifier == "onesvm":
        save_predicted_kmers(bacteria, pd.Series(range(len(ids))), kmers_list, ids, X, bacteria_kmers_file, "binary")
        save_predicted_kmers(unclassified, pd.Series(range(len(ids))), kmers_list, ids, X, unclassified_kmers_file, "binary")
    elif classifier == "linearsvm":
        save_predicted_kmers(bacteria, pd.Series(range(len(ids))), kmers_list, ids, X, bacteria_kmers_file, "binary")
        save_predicted_kmers(host, pd.Series(range(len(ids))), kmers_list, ids, X, host_kmers_file, "binary")
    else:
        save_predicted_kmers(bacteria, pd.Series(range(len(ids))), kmers_list, ids, X, bacteria_kmers_file, "binary")
        save_predicted_kmers(host, pd.Series(range(len(ids))), kmers_list, ids, X, host_kmers_file, "binary")
        save_predicted_kmers(unclassified, pd.Series(range(len(ids))), kmers_list, ids, X, unclassified_kmers_file, "binary")

    classified_data["bacteria"] = {}
    classified_data["bacteria"]["X"] = str(bacteria_kmers_file)
    classified_data["bacteria"]["kmers_list"] = kmers_list
    classified_data["bacteria"]["ids"] = [ids[i] for i in bacteria]

    classified_data["unclassified"] = {}
    classified_data["unclassified"]["X"] = str(unclassified_kmers_file)
    classified_data["unclassified"]["kmers_list"] = kmers_list
    classified_data["unclassified"]["ids"] = [ids[i] for i in unclassified]
    classified_data["order"].append("unclassified")

    classified_data["host"] = {}
    classified_data["host"]["X"] = str(host_kmers_file)
    classified_data["host"]["kmers_list"] = kmers_list
    classified_data["host"]["ids"] = [ids[i] for i in host]
    classified_data["order"].append("host")

    return classified_data
