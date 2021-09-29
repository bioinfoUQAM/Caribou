import pandas as pd
import numpy as np

import sys
import os

from sklearn import metrics, model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from data.build_data import load_Xy_data, save_Xy_data

from keras.models import Sequential
from keras.layers import Dense, LSTM

from data.build_data import load_Xy_data, save_Xy_data

from joblib import dump, load

__author__ = "nicolas"

def bacteria_extraction(metagenome_k_mers, database_k_mers, k, prefix, dataset, classifier = "multiSVM", verbose = 1, saving = 1, cv = 1):
    bacteria_kmers_file = prefix + "_K{}_{}_Xy_bacteria_database_{}_data.hdf5.bz2".format(k, classifier, dataset)

    # Load extracted data if already exists or train and extract bacteria depending on chosen method
    if os.path.isfile(bacteria_kmers_file):
        bacteria = load_Xy_data(bacteria_kmers_file)
    else:
        if classifier in ["oneSVM","lof"]:
            bacteria = bacteria_extraction_binary(metagenome_k_mers, database_k_mers, k, prefix, dataset, classifier, verbose, saving)
        else:
            bacteria = bacteria_extraction_multi(metagenome_k_mers, database_k_mers, k, prefix, dataset, classifier, verbose, saving)

    return bacteria

def bacteria_extraction_binary(metagenome_k_mers, database_k_mers, k, prefix, dataset, classifier = "multiSVM", verbose = 1, saving = 1, cv = 1):
    bacteria_kmers_file = prefix + "_K{}_{}_Xy_bacteria_database_{}_data.hdf5.bz2".format(k, classifier, dataset)
    unclassified_kmers_file = prefix + "_K{}_{}_Xy_unclassified_database_{}_data.hdf5.bz2".format(k, classifier, dataset)
    clf_file = prefix + "_K{}_{}_bacteria_binary_classifier_{}_model.joblib".format(k, classifier, dataset)

    # Get training dataset and assign to variables
    X_train = pd.DataFrame(StandardScaler().fit_transform(database_k_mers["X"]), columns = database_k_mers["kmers_list"], index = database_k_mers["ids"])
    y_train = pd.DataFrame(database_k_mers["y"], index = database_k_mers["ids"])
    y_train.loc[y_train[0] == "bacteria"] = 1
    y_train.loc[y_train[0] != 1] = -1

    # If classifier exists load it or train if not
    if os.path.isfile(clf_file):
        clf = load(clf_file)
    else:
        clf = training(X_train, y_train, classifier = classifier, verbose = verbose, cv = cv)
        dump(clf, clf_file)

    # Classify sequences into bacteria / unclassified and return k-mers profiles for bacteria
    bacteria = extract_bacteria_binary(clf, pd.DataFrame(metagenome_k_mers["X"], columns = metagenome_k_mers["kmers_list"], index = metagenome_k_mers["ids"]), bacteria_kmers_file, unclassified_kmers_file, verbose = verbose, saving = saving)

    return bacteria

def bacteria_extraction_multi(metagenome_k_mers, database_k_mers, k, prefix, dataset, classifier = "multiSVM", verbose = 1, saving = 1, cv = 1):
    bacteria_kmers_file = prefix + "_K{}_{}_Xy_bacteria_database_{}_data.hdf5.bz2".format(k, classifier, dataset)
    virus_kmers_file = prefix + "_K{}_{}_Xy_virus_database_{}_data.hdf5.bz2".format(k, classifier, dataset)
    animals_kmers_file = prefix + "_K{}_{}_Xy_animals_database_{}_data.hdf5.bz2".format(k, classifier, dataset)
    plants_kmers_file = prefix + "_K{}_{}_Xy_plants_database_{}_data.hdf5.bz2".format(k, classifier, dataset)
    fungi_kmers_file = prefix + "_K{}_{}_Xy_fungi_database_{}_data.hdf5.bz2".format(k, classifier, dataset)
    protists_kmers_file = prefix + "_K{}_{}_Xy_protists_database_{}_data.hdf5.bz2".format(k, classifier, dataset)
    clf_file = prefix + "_K{}_{}_bacteria_multi_classifier_{}_model.joblib".format(k, classifier, dataset)

    # Get training dataset and assign to variables
    X_train = pd.DataFrame(StandardScaler().fit_transform(database_k_mers["X"]), columns = database_k_mers["kmers_list"], index = database_k_mers["ids"])
    y_train = pd.DataFrame(database_k_mers["y"].astype(np.int32), index = database_k_mers["ids"])

    # If classifier exists load it or train if not
    if os.path.isfile(clf_file):
        clf = load(clf_file)
    else:
        clf = training(X_train, y_train, classifier = classifier, verbose = verbose, cv = cv)
        dump(clf, clf_file)

    # Classify sequences into known classes and return k-mers profiles for bacteria
    bacteria = extract_bacteria_multi(clf, pd.DataFrame(metagenome_k_mers["X"], columns = metagenome_k_mers["kmers_list"], index = metagenome_k_mers["ids"]), bacteria_kmers_file, virus_kmers_file, animals_kmers_file, plants_kmers_file, fungi_kmers_file, protists_kmers_file, verbose = verbose, saving = saving)

    return bacteria

def extract_bacteria_multi(clf, k_mers, bacteria_kmers_file, virus_kmers_file, animals_kmers_file, plants_kmers_file, fungi_kmers_file, protists_kmers_file, verbose = 1, saving = 1):
    if verbose:
        print("Extracting predicted bacteria and identified classes")
    predict = clf.predict(k_mers)
    bacteria = pd.DataFrame(columns = k_mers.columns)
    if saving:
        virus = pd.DataFrame(columns = k_mers.columns)
        animals = pd.DataFrame(columns = k_mers.columns)
        plants = pd.DataFrame(columns = k_mers.columns)
        fungi = pd.DataFrame(columns = k_mers.columns)
        protists = pd.DataFrame(columns = k_mers.columns)
        for i in range(len(predict)):
            if predict[i] == "bacteria":
                bacteria = bacteria.append(pd.DataFrame(k_mers.iloc[i]).transpose())
            elif predict[i] == "virus":
                virus = virus.append(pd.DataFrame(k_mers.iloc[i]).transpose())
            elif predict[i] == "animals":
                animals = animals.append(pd.DataFrame(k_mers.iloc[i]).transpose())
            elif predict[i] == "plants":
                plants = plants.append(pd.DataFrame(k_mers.iloc[i]).transpose())
            elif predict[i] == "fungi":
                fungi = fungi.append(pd.DataFrame(k_mers.iloc[i]).transpose())
            elif predict[i] == "protists":
                protists = protists.append(pd.DataFrame(k_mers.iloc[i]).transpose())
        save_Xy_data(bacteria, bacteria_kmers_file)
        save_Xy_data(virus, virus_kmers_file)
        save_Xy_data(animals, animals_kmers_file)
        save_Xy_data(plants, plants_kmers_file)
        save_Xy_data(fungi, fungi_kmers_file)
        save_Xy_data(protists, protists_kmers_file)
    else:
        for i in range(len(predict)):
            if predict[i] == 1:
                bacteria = bacteria.append(pd.DataFrame(k_mers.iloc[i]).transpose())
        save_Xy_data(bacteria, bacteria_kmers_file)

    return bacteria

def extract_bacteria_binary(clf, k_mers, bacteria_kmers_file, unclassified_kmers_file, verbose = 1, saving = 1):
    if verbose:
        print("Extracting binary predicted bacteria and unknowns")
    predict = clf.predict(k_mers)
    bacteria = pd.DataFrame(columns = k_mers.columns)
    if saving:
        unclassified = pd.DataFrame(columns = k_mers.columns)
        for i in range(len(predict)):
            if predict[i] == 1:
                bacteria = bacteria.append(pd.DataFrame(k_mers.iloc[i]).transpose())
            elif predict[i] == -1:
                unclassified = unclassified.append(pd.DataFrame(k_mers.iloc[i]).transpose())
        save_Xy_data(bacteria, bacteria_kmers_file)
        save_Xy_data(unclassified, unclassified_kmers_file)
    else:
        for i in range(len(predict)):
            if predict[i] == 1:
                bacteria = bacteria.append(pd.DataFrame(k_mers.iloc[i]).transpose())
        save_Xy_data(bacteria, bacteria_kmers_file)

    return bacteria

def training(X_train, y_train, classifier = "multiSVM", verbose = 1, cv = 1):
    if classifier == "oneSVM":
        if verbose:
            print("Training binary bacterial extractor with One Class SVM")
        clf = OneClassSVM(gamma = 'scale', kernel = "rbf", nu = 0.5)
    elif classifier == "lof":
        if verbose:
            print("Training binary bacterial extractor with Local Outlier Factor")
        clf = LocalOutlierFactor(contamination = 0.5, novelty = True, n_jobs = -1)
    elif classifier == "multiSVM":
        if verbose:
            print("Training multiclass bacterial extractor with Linear SVM")
        clf = LinearSVC(dual = False, random_state = 42)
    elif classifier == "forest":
        if verbose:
            print("Training multiclass bacterial extractor with Random Forest")
        clf = RandomForestClassifier(n_estimators = 100, random_state = 42, n_jobs = -1)
    elif classifier == "knn":
        if verbose:
            print("Training multiclass bacterial extractor with K Nearest Neighbors")
        clf = KNeighborsClassifier(random_state = 42, n_jobs = -1)
    elif classifier == "lstm":
        if verbose:
            print("Training multiclass bacterial extractor with Seeker LSTM method")
        clf = build_LSTM()
    else:
        print("Classifier type unknown !!! \n Models implemented at this moment are \n binary extractors :  One Class SVM (oneSVM) and Local Outlier Factor (lof) \n multiclass extractors : Linear SVM (multiSVM), Random forest (forest), KNN clustering (knn) and LSTM RNN (lstm)")
        sys.exit()

    if cv:
        if classifier == "lstm":
            clf.fit(X_train, y_train, epochs=100, batch_size=27)
            y_pred_test = clf.predict_generator(X_test)
        else:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)

        training_cross_validation(y_pred_test, list(y_test[0]), classifier)
    else:
        if classifier == "lstm":
            clf.fit(X_train, y_train, epochs=100, batch_size=27)
        else:
            clf.fit(X_train, y_train)

    return clf

# From Seeker train_model.py line 69
def build_LSTM():
    """Build the LSTM model for sequence classification."""
    # Initialize a sequential model
    model = Sequential()

    # Add LSTM layer
    model.add(LSTM(5, input_shape=(NUC_COUNT, 1000)))

    # Add Dense NN layer
    model.add(Dense(1, activation='tanh'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

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
