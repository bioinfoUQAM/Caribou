import modin.pandas as pd
import numpy as np

import os
import sys
import ray

from utils import *
from models_classes import SklearnModel, KerasTFModel

__author__ = 'Nicolas de Montigny'

__all__ = ['bacterial_classification','classify']

# TODO: FINISH CONVERTING TO CLASSES FOR MODELS
def bacterial_classification(classified_data, database_k_mers, k, outdirs, dataset, training_epochs, classifier = 'lstm_attention', batch_size = 32, threshold = 0.8, verbose = True, cv = True:
    previous_taxa_unclassified = None

    taxas = database_k_mers['taxas'].copy()

    for taxa in taxas:
        train = False
        classified_kmers_file = '{}Xy_classified_{}_K{}_{}_database_{}_data'.format(outdirs['data_dir'], taxa, k, classifier, dataset)
        unclassified_kmers_file = '{}Xy_unclassified_{}_K{}_{}_database_{}_data'.format(outdirs['data_dir'], taxa, k, classifier, dataset)

        if taxa == taxas[-1]:
            classified_data[taxa] = previous_taxa_unclassified
            classified_data['order'].append(taxa)
        else:
            if classifier in ['sgd','svm','mlr','mnb']:
                model = SklearnModel(classifier, dataset, outdirs['models_dir'], outdirs['results_dir'], batch_size, k, taxa, verbose)
            elif classifier in ['lstm_attention','cnn','widecnn']:
                model = KerasTFModel(classifier, dataset, outdirs['models_dir'], outdirs['results_dir'], batch_size, k, taxa, verbose)
            else:
                print('Bacteria classifier type unknown !!!\n\tModels implemented at this moment are :\n\tLinear models :  Ridge regressor (sgd), Linear SVM (svm), Multiple Logistic Regression (mlr)\n\tProbability classifier : Multinomial Bayes (mnb)\n\tNeural networks : Deep hybrid between LSTM and Attention (lstm_attention), CNN (cnn) and Wide CNN (widecnn)')
                sys.exit()

            if not os.path.isfile(model.clf_file):
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
                    # Keep only classes of sequences that were not removed in kmers extraction
                    X_train = ray.data.read_parquet(database_k_mers['profile'])
                    y_train = pd.DataFrame(database_k_mers['classes'], columns = database_k_mers['taxas']).loc[:,taxa].str.lower()
                    y_train = ray.data.from_modin(y_train[y_train['id'].isin(list(X_train.to_modin()['id']))])

                    model.train(X_train, y_train, cv)

                # Classify sequences into taxa and build k-mers profiles for classified and unclassified data
                # Keep previous taxa to reclassify only unclassified reads at a higher taxonomic level
                if previous_taxa_unclassified is None:
                    if verbose:
                        print('Classifying bacteria sequences at {} level'.format(taxa))
                    df = ray.data.read_parquet(classified_data['bacteria']['profile'])
                    classified_data[taxa], previous_taxa_unclassified = classify(df, clf_file, label_encoder, taxa, classified_kmers_file, unclassified_kmers_file, threshold = threshold, verbose = verbose)
                else:
                    if verbose:
                        print('Classifying bacteria sequences at {} level'.format(taxa))
                    classified_data[taxa], previous_taxa_unclassified = classify(previous_taxa_unclassified['profile'], model, threshold, verbose)

                save_Xy_data(classified_data[taxa], classified_kmers_file)
                save_Xy_data(previous_taxa_unclassified, unclassified_kmers_file)
                classified_data['order'].append(taxa)

    return classified_data

def classify(df_file, model, threshold = 0.8, verbose = True):
    if verbose:
        print('Extracting predicted sequences at {} taxonomic level'.format(taxa))

    df = ray.data.read_parquet(df_file)

    classified_data = {}

    pred = model.predict(df, threshold)

    df = df.to_modin()

    # Make sure classes are writen in lowercase
    pred['class'] = pred['class'].str.lower()

    df_classified = df[pred['class'].str.notequals('unknown')]
    df_unclassified = df[pred['class'].str.match('unknown')]

    # Save / add to classified/unclassified data
    try:
        df_classified.to_parquet(classified_kmers_file)
        classified_data['classified'] = {}
        classified_data['classified']['profile'] = str(classified_kmers_file)
    except:
        if verbose:
            print('No classified data at {} taxonomic level, cannot save it to file or add it to classified data'.format(taxa))
    try:
        df_unclassified.to_parquet(unclassified_kmers_file)
        classified_data['unclassified'] = {}
        classified_data['unclassified']['profile'] = str(unclassified_kmers_file)
    except:
        if verbose:
            print('No unclassified data at {} taxonomic level, cannot save it to file or add it to unclassified data'.format(taxa))

    return classified_data, unclassified_data
