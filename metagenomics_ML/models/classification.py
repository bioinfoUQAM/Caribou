import pandas as pd

import sys
import os

from sklearn.linear_model import SGDOneClassSVM, SGDClassifier
from sklearn.cluster import MiniBatchKMeans

from keras.models import load_model

from utils import *
from models.models_utils import *
from models.build_neural_networks import *

from joblib import load

__author__ = "nicolas"

def bacterial_classification(metagenome_k_mers, database_k_mers, k, prefix, dataset, classifier = "deeplstm", batch_size = 32, verbose = 1, cv = 1):
    # RECURSIVITY / LOOP FOR TAXA RANK (FLEX) CLASSIFICATION OF unclassified
    classified_data = {}
    previous_taxa_unclassified = None

    for taxa in database_k_mers["taxas"]:
        classified_kmers_file = "{}_K{}_{}_Xy_classified_{}_database_{}_data.hdf5.bz2".format(prefix, k, classifier, taxa, dataset) # Pandas df en output?
        unclassified_kmers_file = "{}_K{}_{}_Xy_unclassified_{}_database_{}_data.hdf5.bz2".format(prefix, k, classifier, taxa, dataset)

        if classifier in ["ridge","svm","mlr","kmeans","mnb"]:
            clf_file = "{}_K{}_{}_bacteria_identification_classifier_{}_model.joblib".format(prefix, k, classifier, dataset)
        elif classifier in ["lstm_attention","cnn","deepcnn"]:
            clf_file = "{}_K{}_{}_bacteria_identification_classifier_{}_model".format(prefix, k, classifier, dataset)

        # Load extracted data if already exists or train and classify bacteria depending on chosen method and taxonomic rank
        if os.path.isfile(classified_kmers_file) and os.path.isfile(unclassified_kmers_file):
            if verbose:
                print("Bacteria sequences at {} level already classified".format(taxa))
# SEE IF NECESSARY TO LOAD DEPENDING ON RETURN
            classified_data[taxa] = load_Xy_data(classified_kmers_file)
            previous_taxa_unclassified = load_Xy_data(unclassified_kmers_file)
        else:
            if verbose:
                print("Training classifier with bacteria sequences at {} level".format(taxa))
            # Get training dataset and assign to variables
            X_train = database_k_mers["X"]
# PROBABLY NEED TO CONVERT CLASSES TO NUMBERS FOR CLASSIFICATION
            y_train = pd.DataFrame(database_k_mers["y"], index = database_k_mers["ids"], columns = database_k_mers["taxas"]).loc[:,taxa]
            labels_list = np.unique(y_train)
            nb_classes = len(labels_list)

            # If classifier exists load it or train if not
            if os.path.isfile(clf_file) and classifier in ["ridge","svm","mlr","kmeans","mnb"]:
                clf = load(clf_file)
            elif os.path.isdir(clf_file) and classifier in ["lstm_attention","cnn","deepcnn"]:
                clf = load_model(clf_file)
            else:
                clf = training(X_train, y_train, database_k_mers["kmers"],k, database_k_mers["ids"], nb_classes, labels_list, classifier = classifier, batch_size = batch_size, verbose = verbose, cv = cv, clf_file = clf_file)
            # Classify sequences into taxa and build k-mers profiles for classified and unclassified data
            # Keep previous taxa to reclassify only unclassified reads at a higher taxonomic level
            if previous_taxa_unclassified == None:
                classified_data[taxa], previous_taxa_unclassified = classify(clf_file, metagenome_k_mers["X"], metagenome_k_mers["kmers_list"], metagenome_k_mers["ids"], classifier, nb_classes, labels_list, classified_kmers_file, unclassified_kmers_file, verbose = verbose)
            else:
                classified_data[taxa], previous_taxa_unclassified = classify(clf_file, previous_taxa_unclassified["X"], previous_taxa_unclassified["kmers_list"], previous_taxa_unclassified["ids"], classifier, nb_classes, labels_list, classified_kmers_file, unclassified_kmers_file, verbose = verbose)

    return classified_data

def training(X_train, y_train, kmers, k, ids, nb_classes, labels_list, classifier = "ridge", batch_size = 32, verbose = 1, cv = 1, clf_file = None):
    # Model trained in MetaVW
# decision_function
    if classifier == "ridge":
        if verbose:
            print("Training multiclass classifier with Ridge regression and SGD squared loss")
        clf = SGDClassifier(loss = "squared_error", n_jobs = -1, random_state = 42)
# decision_function
    elif classifier == "svm":
        if verbose:
            print("Training multiclass classifier with Linear SVM and SGD hinge loss")
        clf = SGDClassifier(loss = "hinge", n_jobs = -1, random_state = 42)
    elif classifier == "mlr":
        if verbose:
            print("Training multiclass classifier with Multiple Logistic Regression")
# predict_proba
# decision_function
        clf = SGDClassifier(loss = 'log', n_jobs = -1, random_state = 42)
    elif classifier == "kmeans":
        if verbose:
            print("Training multiclass classifier with K Means")
        clf = MiniBatchKMeans(nclusters = nb_classes, batch_size = batch_size, random_state = 42)
    elif classifier == "mnb":
        if verbose:
            print("Training multiclass classifier with Multinomial Naive Bayes")
# predict_proba
        clf = MultinomialNB()
    elif classifier == "lstm_attention":
        if verbose:
            print("Training multiclass classifier based on Deep Neural Network hybrid between LSTM and Attention")
        clf = build_LSTM_attention(k, nb_classes, batch_size)
    elif classifier == "cnn":
        if verbose:
            print("Training multiclass classifier based on CNN Neural Network")
        clf = build_CNN(nb_classes, len(kmers))
    elif classifier == "deepcnn":
        if verbose:
            print("Training multiclass classifier based on Deep CNN Network")
        clf = build_deepCNN(k, batch_size)
    else:
        print("Bacteria classifier type unknown !!!\n\tModels implemented at this moment are :\n\tLinear models :  Ridge regressor (ridge), Linear SVM (svm), Multiple Logistic Regression (mlr)\n\tClustering classifier : K Means (kmeans)\n\tProbability classifier : Multinomial Bayes (mnb)\n\tNeural networks : Hybrid between LSTM and Attention (lstm_attention), CNN (cnn) and Deep CNN (deepcnn)")
        sys.exit()

    if cv:
        clf = fit_predict_cv(X_train, y_train, batch_size, kmers, ids, classifier, labels_list, clf, verbose = verbose, clf_file = clf_file)
    else:
        clf = fit_model(X_train, y_train, batch_size, kmers, ids, classifier, labels_list, clf, verbose = verbose, clf_file = clf_file)

    return clf

# FINISH FUNCTION TO CLASSIFY AND IDENTIFY CLASSES DEPENDING ON
# LABELS TRANSFORMATION REQUIREMENT AND MULTICLASS OUTPUT
def classify(clf_file, X, kmers_list, ids, classifier, nb_classes, labels_list, classified_kmers_file, unclassified_kmers_file, verbose = 1):
    if verbose:
        print("Classifying sequences at {} level".format(taxa))

    predict = model_predict(clf_file, X, kmers_list, ids, classifier, nb_classes, labels_list, verbose = verbose)
    classified = []
    unclassified = []

    for i in range(len(predict)):
        if predict[i] != -1:
            classified.append(i)
        elif predict[i] == -1:
            unclassified.append(i)

    save_predicted_kmers(classified, pd.Series(range(len(ids))), kmers_list, ids, X, classified_kmers_file)
    save_predicted_kmers(unclassified, pd.Series(range(len(ids))), kmers_list, ids, X, unclassified_kmers_file)

    classified_data = {}
    classified_data["X"] = str(classified_kmers_file)
    classified_data["kmers_list"] = kmers_list
    classified_data["ids"] = [ids[i] for i in classified]

    unclassified_data = {}
    unclassified_data["X"] = str(unclassified_kmers_file)
    unclassified_data["kmers_list"] = kmers_list
    unclassified_data["ids"] = [ids[i] for i in unclassified]

    return classified_data, unclassified_data
