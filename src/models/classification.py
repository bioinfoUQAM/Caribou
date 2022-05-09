import pandas as pd
import numpy as np

import os
import sys
import vaex

from utils import *
from models.models_utils import *
from models.build_neural_networks import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

__author__ = 'Nicolas de Montigny'

__all__ = ['bacterial_classification','training','classify']

def bacterial_classification(classified_data, database_k_mers, k, outdirs, dataset, training_epochs, classifier = 'lstm_attention', batch_size = 32, threshold = 0.8, verbose = 1, cv = 1, n_jobs = 1):
    previous_taxa_unclassified = None

    taxas = database_k_mers['taxas'].copy()

    for taxa in taxas:
        train = False
        classified_kmers_file = '{}Xy_classified_{}_K{}_{}_database_{}_data.hdf5'.format(outdirs['data_dir'], taxa, k, classifier, dataset)
        unclassified_kmers_file = '{}Xy_unclassified_{}_K{}_{}_database_{}_data.hdf5'.format(outdirs['data_dir'], taxa, k, classifier, dataset)

        if taxa == taxas[-1]:
            classified_data[taxa] = previous_taxa_unclassified
            classified_data['order'].append(taxa)
        else:
            clf_file = '{}bacteria_identification_classifier_{}_K{}_{}_{}_model.json'.format(outdirs['models_dir'], taxa, k, classifier, dataset)
            labels_file = '{}label_encoding_{}_K{}_{}_{}_model.hdf5'.format(outdirs['models_dir'], taxa, k, classifier, dataset)
            if not os.path.isfile(clf_file):
                train = True

            # Load extracted data if already exists or train and classify bacteria depending on chosen method and taxonomic rank
            if os.path.isfile(classified_kmers_file) and os.path.isfile(unclassified_kmers_file):
                if verbose:
                    print('Bacteria sequences at {} level already classified'.format(taxa))
                classified_data[taxa] = load_Xy_data(classified_kmers_file)
                previous_taxa_unclassified = load_Xy_data(unclassified_kmers_file)
                classified_data['order'].append(taxa)
            else:
                if verbose:
                    print('Training classifier with bacteria sequences at {} level'.format(taxa))
                # If classifier exists load it or train if not
                if train is True:
                    # Get training dataset and assign to variables
                    df = vaex.open(database_k_mers['profile'])
                    classes_train = pd.DataFrame(database_k_mers['classes'], columns = database_k_mers['taxas']).loc[:,taxa]
                    df['classes'] = np.array(classes_train)
                    df = label_encode(df, labels_file)

                    clf_file = training(df, k, outdirs['plots_dir'] if cv else None, training_epochs, classifier = classifier, batch_size = batch_size, verbose = verbose, cv = cv, clf_file = clf_file, n_jobs = n_jobs)

                # Classify sequences into taxa and build k-mers profiles for classified and unclassified data
                # Keep previous taxa to reclassify only unclassified reads at a higher taxonomic level
                if previous_taxa_unclassified is None:
                    if verbose:
                        print('Classifying bacteria sequences at {} level'.format(taxa))
                    df = vaex.open(classified_data['bacteria']['profile'])
                    classified_data[taxa], previous_taxa_unclassified = classify(df, clf_file, labels_file, taxa, classified_kmers_file, unclassified_kmers_file, threshold = threshold, verbose = verbose)
                else:
                    if verbose:
                        print('Classifying bacteria sequences at {} level'.format(taxa))
                    df = vaex.open(previous_taxa_unclassified['profile'])
                    classified_data[taxa], previous_taxa_unclassified = classify(df, clf_file, labels_file, taxa, classified_kmers_file, unclassified_kmers_file, threshold = threshold, verbose = verbose)

                save_Xy_data(classified_data[taxa],classified_kmers_file)
                save_Xy_data(previous_taxa_unclassified, unclassified_kmers_file)
                classified_data['order'].append(taxa)

    return classified_data

def training(df, k, outdir_plots, training_epochs, classifier = 'lstm_attention', batch_size = 32, verbose = 1, cv = 1, clf_file = None, n_jobs = 1):
    nb_classes = len(df.unique('label_encoded_classes'))
    # Model trained in MetaVW
    if classifier == 'sgd':
        if verbose:
            print('Training multiclass classifier with SGD and squared loss function')
        clf = SGDClassifier(loss = 'squared_error', n_jobs = -1, random_state = 42)
    elif classifier == 'svm':
        if verbose:
            print('Training multiclass classifier with Linear SVM and SGD hinge loss')
        clf = SGDClassifier(loss = 'hinge', n_jobs = -1, random_state = 42)
    elif classifier == 'mlr':
        if verbose:
            print('Training multiclass classifier with Multinomial Logistic Regression')
        clf = SGDClassifier(loss = 'log', n_jobs = -1, random_state = 42)
    elif classifier == 'mnb':
        if verbose:
            print('Training multiclass classifier with Multinomial Naive Bayes')
        clf = MultinomialNB()
    elif classifier == 'lstm_attention':
        if verbose:
            print('Training multiclass classifier based on Deep Neural Network hybrid between LSTM and Attention')
        clf = build_LSTM_attention(k, nb_classes, batch_size)
    elif classifier == 'cnn':
        if verbose:
            print('Training multiclass classifier based on CNN Neural Network')
        clf = build_CNN(k, batch_size, nb_classes)
    elif classifier == 'widecnn':
        if verbose:
            print('Training multiclass classifier based on Wide CNN Network')
        clf = build_wideCNN(k, batch_size, nb_classes)
    else:
        print('Bacteria classifier type unknown !!!\n\tModels implemented at this moment are :\n\tLinear models :  Ridge regressor (sgd), Linear SVM (svm), Multiple Logistic Regression (mlr)\n\tProbability classifier : Multinomial Bayes (mnb)\n\tNeural networks : Deep hybrid between LSTM and Attention (lstm_attention), CNN (cnn) and Wide CNN (widecnn)')
        sys.exit()

    if cv:
        clf_file = cross_validation_training(df, batch_size, k, classifier, outdir_plots, clf, training_epochs, cv = cv, verbose = verbose, clf_file = clf_file, n_jobs = n_jobs)
    else:
        fit_model(df, batch_size, classifier, clf, training_epochs, shuffle = True, clf_file = clf_file)

    return clf_file

def classify(df, clf_file, labels_file, taxa, classified_kmers_file, unclassified_kmers_file, threshold = 0.8, verbose = 1):

    classified_data = {}

    df = model_predict(df, clf_file, threshold = threshold)
    labels_encoding = vaex.open(labels_file)
    df = df.join(labels_encoding, how = left, on = 'label_encoded_classes')

    if verbose:
        print('Extracting predicted sequences at {} taxonomic level'.format(taxa))

    # Make sure classes are writen in lowercase
    df['classes'] = df.classes.str.lower()

    df_classified = df[df.classes.str.notequals('unknown')]
    df_classified = df_classified.drop('label_encoded_classes')
    df_unclassified = df[df.classes.str.match('unknown')]
    df_unclassified = df_unclassified.drop(['classes','label_encoded_classes'])

    # Save / add to classified/unclassified data
    try:
        df_classified.export_hdf5(classified_kmers_file)
        classified_data['classified'] = {}
        classified_data['classified']['profile'] = str(classified_kmers_file)
    except:
        if verbose:
            print('No classified data at {} taxonomic level, cannot save it to file or add it to classified data'.format(taxa))
    try:
        df_unclassified.export_hdf5(unclassified_kmers_file)
        classified_data['unclassified'] = {}
        classified_data['unclassified']['profile'] = str(unclassified_kmers_file)
    except:
        if verbose:
            print('No unclassified data at {} taxonomic level, cannot save it to file or add it to unclassified data'.format(taxa))

    return classified_data, unclassified_data
