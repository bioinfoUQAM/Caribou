import os
import sys
import ray

import modin.pandas as pd

from models.models_classes import SklearnModel, KerasTFModel
from utils import load_Xy_data, save_Xy_data, merge_database_host

__author__ = 'Nicolas de Montigny'

__all__ = ['bacteria_extraction','extract']

def bacteria_extraction(metagenome_k_mers, database_k_mers, k, outdirs, dataset, training_epochs = 100, classifier = 'deeplstm', batch_size = 32, verbose = True, cv = True):
    # classified_data is a dictionnary containing data dictionnaries at each classified level:
    # {taxa:{'X':path to ray dataset in parquet format}}
    classified_data = {'order' : ['bacteria','host','unclassified']}
    train = False

    bacteria_kmers_file = '{}Xy_bacteria_database_K{}_{}_{}_data'.format(outdirs['data_dir'], k, classifier, dataset)
    host_kmers_file = '{}Xy_host_database_K{}_{}_{}_data'.format(outdirs['data_dir'], k, classifier, dataset)
    unclassified_kmers_file = '{}Xy_unclassified_database_K{}_{}_{}_data'.format(outdirs['data_dir'], k, classifier, dataset)
    bacteria_data_file = '{}Xy_bacteria_database_K{}_{}_{}_data.npz'.format(outdirs['data_dir'], k, classifier, dataset)

    if classifier in ['onesvm','linearsvm']:
        model = SklearnModel(classifier, dataset, outdirs['models_dir'], outdirs['results_dir'], batch_size, k, 'domain', verbose)
    elif classifier in ['attention','lstm','deeplstm']:
        model = KerasTFModel(classifier, dataset, outdirs['models_dir'], outdirs['results_dir'], batch_size, training_epochs, k, 'domain', verbose)
    else:
        print('Bacteria extractor unknown !!!\n\tModels implemented at this moment are :\n\tBacteria isolator :  One Class SVM (onesvm)\n\tBacteria/host classifiers : Linear SVM (linearsvm)\n\tNeural networks : Attention (attention), Shallow LSTM (lstm) and Deep LSTM (deeplstm)')
        sys.exit()

    if not os.path.isfile(model.clf_file):
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
        # Keep only classes of sequences that were not removed in kmers extraction
        if classifier == 'onesvm' and isinstance(database_k_mers, tuple):
            print('Classifier One Class SVM cannot be used with host data!\nEither remove host data from config file or choose another bacteria extraction method.')
            sys.exit()
        elif classifier == 'onesvm' and not isinstance(database_k_mers, tuple):
            X_train = ray.data.read_parquet(database_k_mers['profile'])
            y_train = ray.data.from_modin(pd.DataFrame(database_k_mers['classes'], columns = database_k_mers['taxas']).loc[:,'domain'].str.lower())
        elif classifier != 'onesvm' and isinstance(database_k_mers, tuple):
            database_k_mers = merge_database_host(database_k_mers[0], database_k_mers[1])
            X_train = ray.data.read_parquet(database_k_mers['profile'])
            y_train = ray.data.from_modin(pd.DataFrame(database_k_mers['classes'], columns = database_k_mers['taxas']).loc[:,'domain'].str.lower())
        else:
            print('Only classifier One Class SVM can be used without host data!\nEither add host data in config file or choose classifier One Class SVM.')
            sys.exit()

        if train is True:
            print('train == True')
            model.train(X_train, y_train, cv)

        # Classify sequences into bacteria / unclassified / host and build k-mers profiles for bacteria
        if metagenome_k_mers is not None:
            classified_data['bacteria'] = extract(metagenome_k_mers['profile'], model, verbose)
            save_Xy_data(classified_data['bacteria'], bacteria_data_file)

    return classified_data

def extract(df_file, model, verbose = True):
    if verbose:
        print('Extracting predicted bacteria sequences')

    df = ray.data.read_parquet(df_file)

    classified_data = {}

    pred = model.predict(df)

    df = df.to_modin()

    # Make sure classes are writen in lowercase
    pred['class'] = pred['class'].str.lower()

    df_bacteria = df[pred['class'].str.match('bacteria')]
    df_host = df[pred['class'].str.match('host')]
    df_unclassified = df[pred['class'].str.match('unknown')]

    # Save / add to classified data
    try:
        df_bacteria.to_parquet(bacteria_kmers_file)
        classified_data['bacteria'] = {}
        classified_data['bacteria']['profile'] = str(bacteria_kmers_file)
    except:
        print('No bacteria data identified, cannot save it to file or add it to classified data')
    try:
        df_unclassified.to_parquet(unclassified_kmers_file)
        classified_data['unclassified'] = {}
        classified_data['unclassified']['profile'] = str(unclassified_kmers_file)
    except:
        print('No unclassified data identified, cannot save it to file or add it to unclassified data')
    try:
        df_host.to_parquet(host_kmers_file)
        classified_data['host'] = {}
        classified_data['host']['profile'] = str(host_kmers_file)
    except:
        print('No host data identified, cannot save it to file or add it to classified data')

    return classified_data
