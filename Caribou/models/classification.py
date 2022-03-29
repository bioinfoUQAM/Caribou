import numpy as np
import pandas as pd

import sys
import os

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from keras.models import load_model

from Caribou.utils import *
from Caribou.models.models_utils import *
from Caribou.models.build_neural_networks import *

from joblib import load

__author__ = "Nicolas de Montigny"

__all__ = ['bacterial_classification','training','classify']

def bacterial_classification(classified_data, database_k_mers, k, outdirs, dataset, training_epochs, classifier = "lstm_attention", batch_size = 32, threshold = 0.8, verbose = 1, cv = 1, n_jobs = 1):
    if classified_data is not None:
        metagenome_k_mers = classified_data["bacteria"]
        previous_taxa_unclassified = None

    taxas = database_k_mers["taxas"].copy()

    for taxa in taxas:
        train = False
        classified_kmers_file = "{}Xy_classified_{}_K{}_{}_database_{}_data.hdf5".format(outdirs["data_dir"], taxa, k, classifier, dataset)
        unclassified_kmers_file = "{}Xy_unclassified_{}_K{}_{}_database_{}_data.hdf5".format(outdirs["data_dir"], taxa, k, classifier, dataset)

        if taxa == taxas[-1]:
            classified_data[taxa] = previous_taxa_unclassified
            classified_data["order"].append(taxa)
        else:
            if classifier in ["sgd","svm","mlr","mnb"]:
                clf_file = "{}bacteria_identification_classifier_{}_K{}_{}_{}_model.jb".format(outdirs["models_dir"], taxa, k, classifier, dataset)
                if not os.path.isfile(clf_file):
                    train = True

            elif classifier in ["lstm_attention","cnn","widecnn"]:
                clf_file = "{}bacteria_identification_classifier_{}_K{}_{}_{}_model".format(outdirs["models_dir"], taxa, k, classifier, dataset)
                if not os.path.isdir(clf_file):
                    train = True

            # Load extracted data if already exists or train and classify bacteria depending on chosen method and taxonomic rank
            if os.path.isfile(classified_kmers_file) and os.path.isfile(unclassified_kmers_file):
                if verbose:
                    print("Bacteria sequences at {} level already classified".format(taxa))
                classified_data[taxa] = load_Xy_data(classified_kmers_file)
                previous_taxa_unclassified = load_Xy_data(unclassified_kmers_file)
                classified_data["order"].append(taxa)
            else:
                if verbose:
                    print("Training classifier with bacteria sequences at {} level".format(taxa))
                # Get training dataset and assign to variables
                X_train = database_k_mers["X"]
                y_train = pd.DataFrame(database_k_mers["y"], index = database_k_mers["ids"], columns = database_k_mers["taxas"]).loc[:,taxa]
                labels_list_str = np.unique(y_train)
                nb_classes = len(labels_list_str)
                y_train = to_int_cls(y_train, nb_classes, labels_list_str)
                labels_list_int = np.unique(y_train)

                # If classifier exists load it or train if not
                if train is True:
                    clf_file = training(X_train, y_train, database_k_mers["kmers_list"],k, database_k_mers["ids"], nb_classes, labels_list_int, outdirs["plots_dir"] if cv else None, training_epochs, classifier = classifier, batch_size = batch_size, threshold = threshold, verbose = verbose, cv = cv, clf_file = clf_file, n_jobs = n_jobs)
                if classified_data is not None:
                    # Classify sequences into taxa and build k-mers profiles for classified and unclassified data
                    # Keep previous taxa to reclassify only unclassified reads at a higher taxonomic level
                    if previous_taxa_unclassified is None:
                        if verbose:
                            print("Classifying bacteria sequences at {} level".format(taxa))
                        classified_data[taxa], previous_taxa_unclassified = classify(clf_file, metagenome_k_mers["X"], metagenome_k_mers["kmers_list"], metagenome_k_mers["ids"], classifier, nb_classes, labels_list_int, labels_list_str, classified_kmers_file, unclassified_kmers_file, threshold = threshold, verbose = verbose)
                    else:
                        if verbose:
                            print("Classifying bacteria sequences at {} level".format(taxa))
                        classified_data[taxa], previous_taxa_unclassified = classify(clf_file, previous_taxa_unclassified["X"], previous_taxa_unclassified["kmers_list"], previous_taxa_unclassified["ids"], classifier, nb_classes, labels_list_int, labels_list_str, classified_kmers_file, unclassified_kmers_file, threshold = threshold, verbose = verbose)
            if classified_data is not None:
                save_Xy_data(classified_data[taxa],classified_kmers_file)
                save_Xy_data(previous_taxa_unclassified, unclassified_kmers_file)
                classified_data["order"].append(taxa)

    if classified_data is not None:
        return classified_data

def training(X_train, y_train, kmers, k, ids, nb_classes, labels_list, outdir_plots, training_epochs, classifier = "lstm_attention", batch_size = 32, threshold = 0.8, verbose = 1, cv = 1, clf_file = None, n_jobs = 1):
    # Model trained in MetaVW
    if classifier == "sgd":
        if verbose:
            print("Training multiclass classifier with SGD and squared loss function")
        clf = SGDClassifier(loss = "squared_error", n_jobs = -1, random_state = 42)
    elif classifier == "svm":
        if verbose:
            print("Training multiclass classifier with Linear SVM and SGD hinge loss")
        clf = SGDClassifier(loss = "hinge", n_jobs = -1, random_state = 42)
    elif classifier == "mlr":
        if verbose:
            print("Training multiclass classifier with Multinomial Logistic Regression")
        clf = SGDClassifier(loss = 'log', n_jobs = -1, random_state = 42)
    elif classifier == "mnb":
        if verbose:
            print("Training multiclass classifier with Multinomial Naive Bayes")
        clf = MultinomialNB()
    elif classifier == "lstm_attention":
        if verbose:
            print("Training multiclass classifier based on Deep Neural Network hybrid between LSTM and Attention")
        clf = build_LSTM_attention(k, nb_classes, batch_size)
    elif classifier == "cnn":
        if verbose:
            print("Training multiclass classifier based on CNN Neural Network")
        clf = build_CNN(k, batch_size, nb_classes)
    elif classifier == "widecnn":
        if verbose:
            print("Training multiclass classifier based on Wide CNN Network")
        clf = build_wideCNN(k, batch_size, nb_classes)
    else:
        print("Bacteria classifier type unknown !!!\n\tModels implemented at this moment are :\n\tLinear models :  Ridge regressor (sgd), Linear SVM (svm), Multiple Logistic Regression (mlr)\n\tProbability classifier : Multinomial Bayes (mnb)\n\tNeural networks : Deep hybrid between LSTM and Attention (lstm_attention), CNN (cnn) and Wide CNN (widecnn)")
        sys.exit()

    if cv:
        clf_file = cross_validation_training(X_train, y_train, batch_size, kmers, ids, classifier, labels_list, outdir_plots, clf, training_epochs, threshold = threshold, verbose = verbose, clf_file = clf_file, n_jobs = n_jobs)
    else:
        fit_model(X_train, y_train, batch_size, kmers, ids, classifier, labels_list, clf, training_epochs, verbose = verbose, clf_file = clf_file)

    return clf_file

def classify(clf_file, X, kmers_list, ids, classifier, nb_classes, labels_list_int, labels_list_str, classified_kmers_file, unclassified_kmers_file, threshold = 0.8, verbose = 1):

    predict = model_predict(clf_file, X, kmers_list, ids, classifier, nb_classes, labels_list_int, threshold = threshold, verbose = verbose)
    classified = []
    unclassified = []

    for i in range(len(predict)):
        if predict[i] != -1:
            classified.append(i)
        elif predict[i] == -1:
            unclassified.append(i)

    save_predicted_kmers(classified, pd.Series(range(len(ids))), kmers_list, ids, X, classified_kmers_file, "multi")
    save_predicted_kmers(unclassified, pd.Series(range(len(ids))), kmers_list, ids, X, unclassified_kmers_file, "multi")

    classified_data = {}
    classified_data["X"] = str(classified_kmers_file)
    classified_data["kmers_list"] = kmers_list
    classified_data["ids"] = [ids[i] for i in classified]
    classified_data["classification"] = [predict[i] for i in classified]

    unclassified_data = {}
    unclassified_data["X"] = str(unclassified_kmers_file)
    unclassified_data["kmers_list"] = kmers_list
    unclassified_data["ids"] = [ids[i] for i in unclassified]

    return classified_data, unclassified_data
