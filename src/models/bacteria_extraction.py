import numpy as np
import modin.pandas as pd

import os
import sys

from utils import *
from models.models_utils import *
from models.build_neural_networks import *
from sklearn.linear_model import SGDOneClassSVM, SGDClassifier

__author__ = 'Nicolas de Montigny'

__all__ = ['bacteria_extraction','training','extract_bacteria_sequences']

def bacteria_extraction(metagenome_k_mers, database_k_mers, k, outdirs, dataset, training_epochs, classifier = 'deeplstm', batch_size = 32, verbose = 1, cv = 1, n_jobs = 1):
    # classified_data is a dictionnary containing data dictionnaries at each classified level:
    # {taxa:{'X':path to vaex dataframe hdf5 file}}
    classified_data = {'order' : ['bacteria','host','unclassified']}
    train = False

    bacteria_kmers_file = '{}Xy_bacteria_database_K{}_{}_{}_data.parquet'.format(outdirs['data_dir'], k, classifier, dataset)
    host_kmers_file = '{}Xy_host_database_K{}_{}_{}_data.parquet'.format(outdirs['data_dir'], k, classifier, dataset)
    unclassified_kmers_file = '{}Xy_unclassified_database_K{}_{}_{}_data.parquet'.format(outdirs['data_dir'], k, classifier, dataset)
    bacteria_data_file = '{}Xy_bacteria_database_K{}_{}_{}_data.npz'.format(outdirs['data_dir'], k, classifier, dataset)

    if classifier in ['onesvm','linearsvm']:
        clf_file = '{}bacteria_binary_classifier_K{}_{}_{}_model.jb'.format(outdirs['models_dir'], k, classifier, dataset)
    else:
        clf_file = '{}bacteria_binary_classifier_K{}_{}_{}_model'.format(outdirs['models_dir'], k, classifier, dataset)

    if not os.path.isfile(clf_file):
        train = True

    # Load extracted data if already exists or train and extract bacteria depending on chosen method
    if os.path.isfile(bacteria_data_file):
        classified_data['bacteria'] = load_Xy_data(bacteria_data_file)
        try:
            classified_data['host'] = load_Xy_data(host_data_file)
        except:
            pass
        classified_data['unclassified'] = load_Xy_data(unclassified_data_file)
        if verbose:
            print('Bacteria sequences already extracted. Skipping this step')
    else:
        # Get training dataset and assign to variables
        if classifier == 'onesvm' and isinstance(database_k_mers, tuple):
            print('Classifier One Class SVM cannot be used with host data!\nEither remove host data from config file or choose another bacteria extraction method.')
            sys.exit()
        elif classifier == 'onesvm' and not isinstance(database_k_mers, tuple):
            X_train = pd.read_parquet(database_k_mers['profile'])
            y_train = pd.DataFrame(database_k_mers['classes'], columns = database_k_mers['taxas']).loc[:,'domain']
            y_train, label_encoder = label_encode(y_train)
        elif classifier != 'onesvm' and isinstance(database_k_mers, tuple):
            database_k_mers = merge_database_host(database_k_mers[0], database_k_mers[1])
            X_train = pd.read_parquet(database_k_mers['profile'])
            y_train = pd.DataFrame(database_k_mers['classes'], columns = database_k_mers['taxas']).loc[:,'domain'].str.lower()
            y_train, label_encoder = label_encode(y_train)
        else:
            print('Only classifier One Class SVM can be used without host data!\nEither add host data in config file or choose classifier One Class SVM.')
            sys.exit()

        # If classifier exists load it or train if not
        if train is True:
            clf_file = training(X_train, y_train, k, outdirs['plots_dir'] if cv else None, training_epochs, classifier = classifier, batch_size = batch_size, verbose = verbose, cv = cv, clf_file = clf_file, n_jobs = n_jobs)

        # Classify sequences into bacteria / unclassified / host and build k-mers profiles for bacteria
        if metagenome_k_mers is not None:
            df = pd.read_parquet(metagenome_k_mers['profile'])
            classified_data['bacteria'] = extract_bacteria_sequences(df, clf_file, label_encoder, bacteria_kmers_file, host_kmers_file, unclassified_kmers_file, verbose = verbose)
            save_Xy_data(classified_data['bacteria'], bacteria_data_file)

            return classified_data


def training(X_train, y_train, k, outdir_plots, training_epochs, classifier = 'deeplstm', batch_size = 32, verbose = 1, cv = 1, clf_file = None, n_jobs = 1):
    if classifier == 'onesvm':
        if verbose:
            print('Training bacterial extractor with One Class SVM')
        clf = SGDOneClassSVM(nu = 0.05, tol = 1e-4)
    elif classifier == 'linearsvm':
        if verbose:
            print('Training bacterial / host classifier with Linear SVM')
        clf = SGDClassifier(early_stopping = False, n_jobs = -1)
    elif classifier == 'attention':
        if verbose:
            print('Training bacterial / host classifier based on Attention Weighted Neural Network')
        clf = build_attention(k)
    elif classifier == 'lstm':
        if verbose:
            print('Training bacterial / host classifier based on Shallow LSTM Neural Network')
        clf = build_LSTM(k, batch_size)
    elif classifier == 'deeplstm':
        if verbose:
            print('Training bacterial / host classifier based on Deep LSTM Neural Network')
        clf = build_deepLSTM(k, batch_size)
    else:
        print('Bacteria extractor unknown !!!\n\tModels implemented at this moment are :\n\tBacteria isolator :  One Class SVM (onesvm)\n\tBacteria/host classifiers : Linear SVM (linearsvm)\n\tNeural networks : Attention (attention), Shallow LSTM (lstm) and Deep LSTM (deeplstm)')
        sys.exit()

    if cv:
        clf_file = cross_validation_training(X_train, y_train, batch_size, k, classifier, outdir_plots, clf, training_epochs, cv = cv, verbose = verbose, clf_file = clf_file, n_jobs = n_jobs)
    else:
        fit_model(X_train, y_train, batch_size, classifier, clf, training_epochs, shuffle = True, clf_file = clf_file)

    return clf_file

def extract_bacteria_sequences(df, clf_file, label_encoder, bacteria_kmers_file, host_kmers_file, unclassified_kmers_file = None, verbose = 1):

    classified_data = {}

    df = model_predict(df, clf_file)
    df = label_decode(df, label_encoder)

    if verbose:
        print('Extracting predicted bacteria sequences')

    # Make sure classes are writen in lowercase
    df['classes'] = df['classes'].str.lower()

    df_bacteria = df[df['classes'].str.match('bacteria')]
    df_host = df[df['classes'].str.match('host')]
    df_unclassified = df[df['classes'].str.match('unknown')]

    # Save / add to classified data
    try:
        df_bacteria.to_parquet(bacteria_kmers_file)
        classified_data['bacteria'] = {}
        classified_data['bacteria']['profile'] = str(bacteria_kmers_file)
    except:
        if verbose:
            print('No bacteria data identified, cannot save it to file or add it to classified data')
    try:
        df_unclassified.to_parquet(unclassified_kmers_file)
        classified_data['unclassified'] = {}
        classified_data['unclassified']['profile'] = str(unclassified_kmers_file)
    except:
        if verbose:
            print('No unclassified data identified, cannot save it to file or add it to unclassified data')
    try:
        df_host.to_parquet(host_kmers_file)
        classified_data['host'] = {}
        classified_data['host']['profile'] = str(host_kmers_file)
    except:
        if verbose:
            print('No host data identified, cannot save it to file or add it to classified data')

    return classified_data
