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
    for taxa in database_k_mers["taxas"]:
        classified_kmers_file = "{}_K{}_{}_Xy_classified_{}_database_{}_data.hdf5.bz2".format(prefix, k, classifier, taxa, dataset) # Pandas df en output?
        unclassified_kmers_file = "{}_K{}_{}_Xy_unclassified_{}_database_{}_data.hdf5.bz2".format(prefix, k, classifier, taxa, dataset)
# CHANGE NAMES DEPENDING ON CLASSIFIERS USED
        if classifier in ["onesvm","linearsvm"]:
            clf_file = "{}_K{}_{}_bacteria_identification_classifier_{}_model.joblib".format(prefix, k, classifier, dataset)
        else:
            clf_file = "{}_K{}_{}_bacteria_identification_classifier_{}_model".format(prefix, k, classifier, dataset)

        if verbose:
            print("Training classifier with bacteria sequences at {} level".format(taxa))

        # Load extracted data if already exists or train and classify bacteria depending on chosen method and taxonomic rank
        if os.path.isfile(classified_kmers_file) and os.path.isfile(unclassified_kmers_file):
            classified = load_Xy_data(classified_kmers_file)
            unclassified = load_Xy_data(unclassified_kmers_file)
        else:
            # Get training dataset and assign to variables
            X_train = database_k_mers["X"]
# PROBABLY NEED TO CONVERT CLASSES TO NUMBERS FOR CLASSIFICATION
            y_train = pd.DataFrame(database_k_mers["y"], index = database_k_mers["ids"], columns = database_k_mers["taxas"]).loc[:,taxa]

        # If classifier exists load it or train if not
# CHANGE NAMES DEPENDING ON CLASSIFIERS USED
        if os.path.isfile(clf_file) and classifier in ["onesvm","linearsvm"]:
            clf = load(clf_file)
# CHANGE NAMES DEPENDING ON CLASSIFIERS USED
        elif os.path.isfile(clf_file) and classifier in ["attention","lstm","deeplstm"]:
            clf = load_model(clf_file)
        else:
            clf = training()

        # Classify sequences into taxa and build k-mers profiles classified data
        classified = extract_bacteria_sequences()

# MAYBE SEND DIRECTLY TO OUTPUTING FUNCTIONS

# NOT SURE IF RETURN CLASSIFIED DATA OR ONLY SAVED TO DISK
    return classified

def training(X_train, y_train, kmers, k, ids, classifier = "ridge", batch_size = 32, verbose = 1, cv = 1, clf_file = None):
# NEED TO TUNE HYPERPARAMETERS
    if classifier == "ridge":
        if verbose:
            print("Training multiclass classifier with Ridge regression and SGD squared loss")
        clf = SGDClassifier(loss = "squared_error", n_jobs = -1, random_state = 42)
    elif classifier == "svm":
        if verbose:
            print("Training multiclass classifier with Linear SVM and SGD hinge loss")
        clf = SGDClassifier(loss = "hinge", n_jobs = -1, random_state = 42)
    elif classifier == "mlr":
        if verbose:
            print("Training multiclass classifier with Multiple Logistic Regression")
# PROBABILITY
        clf = SGDClassifier(loss = 'log', n_jobs = -1, random_state = 42)
    elif classifier == "kmeans":
        if verbose:
            print("Training multiclass classifier with K Means")
        clf = MiniBatchKMeans(nclusters = len(np.unique(y_train)), batch_size = batch_size, random_state = 42)
    elif classifier == "mnb":
        if verbose:
            print("Training multiclass classifier with Multinomial Naive Bayes")
# PROBABILITY
        clf = MultinomialNB()
    elif classifier == "lstm_attention":
        if verbose:
            print("Training multiclass classifier based on Deep Neural Network hybrid between LSTM and Attention")
        clf = build_LSTM_attention()
    elif classifier == "cnn":
        if verbose:
            print("Training multiclass classifier based on CNN Neural Network")
        clf = build_CNN()
    elif classifier == "dbn":
        if verbose:
            print("Training multiclass classifier based on Deep DBN Neural Network")
        clf = build_DBN()
    elif classifier == "deepcnn":
        if verbose:
            print("Training multiclass classifier based on Deep CNN Network")
        clf = build_WDcnn()
    else:
        print("Bacteria classifier type unknown !!!\n\tModels implemented at this moment are :\n\tLinear models :  Ridge regressor (ridge), Linear SVM (svm), Multiple Logistic Regression (mlr)\n\tClustering classifier : K Means (kmeans)\n\tProbability classifier : Multinomial Bayes (mnb)\n\tNeural networks : Hybrid between LSTM and Attention (lstm_attention), CNN (cnn), DBN (dbn) and Deep CNN (deepcnn)")
        sys.exit()

    if cv:
        clf = fit_predict_cv_multi()
    else:
        clf = fit_model_multi()

    return clf

# TO BE UPGRADED FOR DEALING WITH VIARIABLE NUMBER OF LABELS
def classify(clf_file, X, kmers_list, ids, classifier, classified_kmers_file, unclassified_kmers_file, verbose = 1):
# EXCLUDE CLASSIFICATIONS BASED ON CONFIDENCE SCORE AFTER PREDICTION
    if verbose:
        print("Extracting classification at {} level".format(taxa))

    predict = model_predict_multi()
    classified = []
    unclassified = []

    for i in range(len(predict)):
        if predict[i] >= 1:
            bacteria.append(i)
        elif predict[i] == -1:
            unclassified.append(i)

    save_predicted_kmers(classified, pd.Series(range(len(ids))), kmers_list, ids, X, classified_kmers_file)
    save_predicted_kmers(unclassified, pd.Series(range(len(ids))), kmers_list, ids, X, unclassified_kmers_file)

# SEE WHAT TO RETURN DEPENDING ON OUTPUT POSSIBILITIES
