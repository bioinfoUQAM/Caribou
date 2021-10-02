
import pandas as pd
import numpy as np

import sys
import os

from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM, LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB


from utils import training_cross_validation, load_Xy_data, save_Xy_data

from joblib import dump, load

__author__ = "nicolas"

def bacterial_classification(bacteria_k_mers, database_k_mers, k, prefix, dataset, classifier = "svm", verbose = 1):
# RECURSIVITY / LOOP FOR TAXA RANK (FLEX) CLASSIFICATION OF unclassified
# Adapt code to accept taxa changes
#    for taxa in ["species","genus","family","order"]:
    taxa = "species"

    classified_file = prefix + "_K{}_{}_Xy_classified_{}_database_{}_data.hdf5.bz2".format(k, classifier, taxa, dataset) # Pandas df en output?
    unclassified_file = prefix + "_K{}_{}_Xy_unclassified_{}_database_{}_data.hdf5.bz2".format(k, classifier, taxa, dataset)
    clf_file = prefix + "_K{}_{}_bacteria_identification_classifier_{}_model.joblib".format(k, classifier, dataset)

    clf = {}
    classified = pd.DataFrame(columns = ["species","genus","family","order"], index = database_k_mers["ids"])
    unclassified = pd.DataFrame(np.zeros((4, len(database_k_mers["ids"]))), columns = ["species","genus","family","order"], index = database_k_mers["ids"]) # Add 1 if not classified at this taxa

    # Get training dataset
    X_train = pd.DataFrame(StandardScaler().fit_transform(database_k_mers["X"]), columns = database_k_mers["kmers_list"], index = database_k_mers["ids"])
# NEED MORE RANKS FOR Y LABELS -> SPECIFY RANK IN LOOP
# Need to find a way to get them from DB and add to class.csv
    y_train = pd.DataFrame(database_k_mers["y"].astype(np.int32), index = database_k_mers["ids"])

# ADD TAXA RANKS
        # Load classifier if already trained
        if os.path.isfile(clf_file):
            clf = load(clf_file)
        else:
            # Train/test classifier
            clf = training(X_train, y_train, classifier = classifier, verbose = verbose)
            dump(clf, clf_file)

        # Classify sequences into taxa / unclassified and return k-mers profiles + classification
        classified, unclassified = classify(clf, k_mers, taxa, classified, unclassified, classified_file, unclassified_file, verbose = verbose, saving = saving)

    return classified


# If class is really unknown and can be anything not from three defined classes, then you better look at anomaly detection methods.
# For example, for your each incoming data first classify if it is anomaly and then if it is not classify it into three well-defined classes.
# Pretty much universal and starting method for anomaly detection is one class SVM.
def training(X_train, y_train, classifier = "multiSVM", verbose = 1):
# NEED TO TUNE HYPERPARAMETERS
    if classifier == "metaVW":
        if verbose:
            print("Training multiclass classifier with Ridge regression and SGD squared loss")
        #clf = Ridge(random_state = 42)
        clf = SGDClassifier(loss = "squared_error", n_jobs = -1, random_state = 42)
    elif classifier == "multiSVM":
        if verbose:
            print("Training multiclass classifier with Linear SVM and SGD hinge loss")
        #clf = LinearSVC(dual = False, random_state = 42)
        clf = SGDClassifier(loss = "hinge", n_jobs = -1, random_state = 42)
    elif classifier == "forest":
        if verbose:
            print("Training multiclass classifier with Random Forest")
        clf = RandomForestClassifier(n_estimators = 100, random_state = 42, n_jobs = -1)
    elif classifier == "knn":
        if verbose:
            print("Training multiclass classifier with K Nearest Neighbors")
        clf = KNeighborsClassifier(random_state = 42, n_jobs = -1)
    elif classifier == "NB":
        if verbose:
            print("Training multiclass classifier with Multinomial Naive Bayes")
        clf = MultinomialNB()
    elif classifier == "regression":
        if verbose:
            print("Training multiclass classifier with Logistic Regression")
        clf = LogisticRegression(penalty = "l2", solver = "saga", multi_class = "multinomial", n_jobs = -1, random_state = 42)
    elif classifier == "bag":
        if verbose:
            print("Training multiclass classifier with K Nearest Neighbors with bagging ensemble method")
        clf = BaggingClassifier(KNeighborsClassifier(), random_state = 42, n_jobs = -1)
    elif classifier == "boost":
        if verbose:
            print("Training multiclass classifier with K Nearest Neighbors with boosting ensemble method")
        clf = AdaBoostClassifier(KNeighborsClassifier(), random_state = 42)
    elif classifier == "consensus":
        if verbose:
            print("Training voting multiclass classifier with all other available classifiers")
# USE TUNED HYPERPARAMETERS FROM OTHER TECHNIQUES
        clf = VotingClassifier(estimators = [("metaVW",SGDClassifier(loss = "squared_error", random_state = 42)),
                                            ("multiSVM",SGDClassifier(loss = "hinge", random_state = 42)),
                                            ("forest",RandomForestClassifier(n_estimators = 100, random_state = 42)),
                                            ("knn",KNeighborsClassifier(random_state = 42)),
                                            ("NB",MultinomialNB()),
                                            ("regression",LogisticRegression(penalty = "l2", solver = "saga", multi_class = "multinomial", random_state = 42)),
                                            ("bagging",BaggingClassifier(KNeighborsClassifier(), random_state = 42)),
                                            ("boosting",AdaBoostClassifier(KNeighborsClassifier(), random_state = 42))],
                                            voting = "soft",
                                            n_jobs = -1)
#    elif classifier == "NN":
#        print("possibility of using neural network from articles/invention")
    else:
        print("Classifier type unknown !!! \n Models implemented at this moment are \n :  MetaVW (metaVW), Linear SVM (multiSVM), Random forest (forest), \nKNN clustering (knn), bagging (bag), Adaboost (boost), Multinomial Bayes (NB), Linear regression (regression) and Multiclassifiers concensus (consensus)")
        sys.exit()

    if cv:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        training_cross_validation(y_pred_test, list(y_test[0]), classifier)

    else:
        clf.fit(X_train, y_train)

    return clf

def classify(clf, k_mers, taxa, classified, unclassified, classified_file, unclassified_file, verbose = 1, saving = 1):
    if verbose:
        print("Extracting classification at {} level".format(taxa))

# À voir comment ajouter outliers detection avant pour prédire les classes inconnues
    predict = clf.predict(k_mers)

    if saving:
# À voir comment mettre ds pandas DF et si px valider avec les labels ou si list fonctionne / options output
        classified[taxa] = predict
        for i in range(len(predict)):
            if predict[i] == -1:
                unclassified.loc[:,taxa].iloc[i,:] = 1

        save_Xy_data(classified, classified_file)
        save_Xy_data(unclassified, unclassified_file)

    else:
# À voir comment mettre ds pandas DF et si px valider avec les labels ou si list fonctionne / options output
        classified[taxa] = predict
        save_Xy_data(classified, classified_file)

    return classified, unclassified
