#!/usr/bin python3

import os
import ray
import sys
import json
import logging
import warnings
import argparse
import numpy as np
import pandas as pd

from glob import glob
from pathlib import Path
from shutil import rmtree

# Package modules
from utils import *
from models.reads_simulation import readsSimulation
from models.ray_tensor_min_max import TensorMinMaxScaler
from models.sklearn.ray_sklearn_partial_trainer import SklearnPartialTrainer
from models.sklearn.ray_sklearn_onesvm_encoder import OneClassSVMLabelEncoder

# Preprocessing
from ray.data.preprocessors import Chain, LabelEncoder

# Training
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDOneClassSVM, SGDClassifier

# Tuning
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig, ScalingConfig
from ray.tune.search.basic_variant import BasicVariantGenerator

warnings.simplefilter(action='ignore')
# Functions
################################################################################

def load_data(opt):
    print('load_data')
    if opt['data_host']:
        data_db = load_Xy_data(opt['data'])
        data_host = load_Xy_data(opt['data_host'])
        data = merge_database_host(data_db, data_host)
        datasets, preprocessor, labels_map = load_ds(
            data,
            f'merged_{opt["database_name"]}_{opt["host_name"]}',
            opt['taxa'],
            data['kmers'],
            opt['kmers_length'],
            opt['classifier']
        )
    else:
        data = load_Xy_data(opt['data'])
        datasets, preprocessor, labels_map = load_ds(
            data,
            opt['database_name'],
            opt['taxa'],
            data['kmers'],
            opt['kmers_length'],
            opt['classifier']
        )

    return data, datasets, preprocessor, labels_map

# Function from class function models.classification.ClassificationMethods._load_training_data


def load_ds(db_data, db_name, taxa, cols, klen, classifier):
    print('load_ds')
    X_train = ray.data.read_parquet(db_data['profile'])
    y_train = pd.DataFrame(
        db_data['classes'],
        columns = db_data['taxas']
    )
    
    for col in y_train.columns:
        y_train[col] = y_train[col].str.lower()
    
    if 'domain' in y_train.columns:
        y_train[y_train['domain'] == 'archaea'] = 'bacteria'

    df_train = zip_X_y(X_train, y_train)
    df_val, df_test = generate_val_test_ds(df_train, db_data, db_name, taxa, cols, klen)
    
    preprocessor, labels_map = preprocess(df_train, taxa, cols, classifier)

    datasets = {
        'train': ray.put(df_train),
        'validation': ray.put(df_val),
        'test': ray.put(df_test)
    }

    return datasets, preprocessor, labels_map


def generate_val_test_ds(df_train, db_data, db_name, taxa, cols, klen):
    print('generate_val_test_ds')
    df_val = df_train.random_sample(0.3)
    df_val = sim_4_cv(df_val, db_data, f'{db_name}_val', taxa, cols, klen)
    df_test = df_train.random_sample(0.3)
    df_test = sim_4_cv(df_test, db_data, f'{db_name}_test', taxa, cols, klen)
    
    return df_val, df_test

# Function from class function models.classification.ClassificationMethods._merge_database_host
def merge_database_host(database_data, host_data):
    print('merge_database_host')
    merged_database_host = {}

    merged_database_host['profile'] = '{}_host_merged'.format(os.path.splitext(database_data['profile'])[0]) # Kmers profile

    df_classes = pd.DataFrame(database_data['classes'], columns=database_data['taxas'])
    df_cls_host = pd.DataFrame(host_data['classes'], columns=host_data['taxas'])

    if len(np.unique(df_classes['domain'])) != 1:
        df_classes[df_classes != 'bacteria'] = 'bacteria'
    
    if len(df_cls_host) > len(host_data['ids']):
        to_remove = np.arange(len(df_cls_host) - len(host_data['ids']))
        df_cls_host.drop(to_remove, axis = 0, inplace = True)
    elif len(df_cls_host) < len(host_data['ids']):
        diff = len(host_data['ids']) - len(df_cls_host)
        row = df_cls_host.iloc[0]
        for i in range(diff):
            df_cls_host = pd.concat([df_cls_host, row.to_frame().T], ignore_index=True)

    df_classes = pd.concat([df_classes, df_cls_host], ignore_index=True)
    merged_database_host['classes'] = np.array(df_classes)  # Class labels
    merged_database_host['ids'] = np.concatenate((database_data['ids'], host_data['ids'])) # IDs
    merged_database_host['kmers'] = database_data['kmers']  # Features
    merged_database_host['taxas'] = database_data['taxas']  # Known taxas for classification
    merged_database_host['fasta'] = (database_data['fasta'], host_data['fasta'])  # Fasta file needed for reads simulation

    df_db = ray.data.read_parquet(database_data['profile'])
    df_host = ray.data.read_parquet(host_data['profile'])
    df_merged = df_db.union(df_host)
    df_merged.write_parquet(merged_database_host['profile'])
    return merged_database_host

# Function from class function models.ray_sklearn.SklearnModel._training_preprocess
def preprocess(df, taxa, cols, classifier):
    print('preprocess')
    if classifier == 'onesvm':
        encoder = OneClassSVMLabelEncoder(taxa)
        encoded = np.array([1,-1], dtype = np.int32)
        labels = np.array(['bacteria', 'unknown'], dtype = object)
    else:
        encoder = LabelEncoder(taxa)

    preprocessor = Chain(
        TensorMinMaxScaler(cols),
        encoder
    )
    preprocessor.fit(df)
    if classifier != 'onesvm':
        labels = list(preprocessor.preprocessors[1].stats_[f'unique_values({taxa})'].keys())
        encoded = np.arange(len(labels))
        labels = np.append(labels, 'unknown')
        encoded = np.append(encoded, -1)
    labels_map = zip(labels, encoded)

    return preprocessor, labels_map

# Function from class function models.classification.ClassificationMethods._sim_4_cv
def sim_4_cv(df, kmers_ds, name, taxa, cols, klen):
    print('_sim_4_cv')
    sim_cls_dct = {
        'id':[],
    }
    taxa_cols = []
    for row in df.iter_rows():
        if len(taxa_cols) == 0:
            taxa_cols = list(row.keys())
            taxa_cols.remove('id')
            taxa_cols.remove('__value__')
            for taxa in taxa_cols:
                sim_cls_dct[taxa] = []
        sim_cls_dct['id'].append(row['id'])
        for taxa in taxa_cols:
            sim_cls_dct[taxa].append(row[taxa])
    cls = pd.DataFrame(sim_cls_dct)
    sim_outdir = os.path.dirname(kmers_ds['profile'])
    cv_sim = readsSimulation(kmers_ds['fasta'], cls, sim_cls_dct['id'], 'miseq', sim_outdir, name)
    sim_data = cv_sim.simulation(klen, cols)
    sim_ids = sim_data['ids']
    sim_cls = pd.DataFrame({'sim_id':sim_ids}, dtype = object)
    sim_cls['id'] = sim_cls['sim_id'].str.replace('_[0-9]+_[0-9]+_[0-9]+', '', regex=True)
    sim_cls = sim_cls.set_index('id').join(cls.set_index('id'))
    sim_cls = sim_cls.drop(['sim_id'], axis=1)
    sim_cls = sim_cls.reset_index(drop = True)
    df = ray.data.read_parquet(sim_data['profile'])
    df = zip_X_y(df, sim_cls)
    return df

def get_clf_params(classifier):
    clf = None
    train_params = {}
    tune_params = {}

    if classifier == 'onesvm':
        clf = SGDOneClassSVM()
        train_params = {
            'nu' : 0.05,
            'tol' : 1e-4
        }
        tune_params = {
            'params' : {
                'nu' : tune.grid_search(np.linspace(0.1, 1, 10)),
                'learning_rate' : tune.grid_search(['constant','optimal','invscaling','adaptive']),
                'eta0' : tune.grid_search(np.logspace(-4,4,10))
            }
        }
    elif classifier == 'linearsvm' or classifier == 'sgd':
        clf = SGDClassifier()
        train_params = {
            'loss' : 'squared_error'
        }
        tune_params = {
            'params' : {
                'loss' : tune.grid_search(['log_loss', 'modified_huber']),
                'penalty' : tune.grid_search(['l2', 'l1', 'elasticnet']),
                'alpha' : tune.grid_search(np.logspace(-4,4,10)),
                'learning_rate' : tune.grid_search(['constant','optimal','invscaling','adaptive']),
                'eta0' : tune.grid_search(np.logspace(-4,4,10))
            }
        }
    elif classifier == 'mnb':
        clf = MultinomialNB()
        train_params = {
            'alpha' : 1.0
        }
        tune_params = {
            'params' : {
                'alpha' : tune.grid_search(np.linspace(0,1,10)),
                'fit_prior' : tune.grid_search([True,False])
            }
        }
    
    return clf, train_params, tune_params


def output_results(opt, tuning_results):
    print('output_results')
    outfile = os.path.join(opt['outdir'], '{}_{}_tuning.csv'.format(opt['classifier'],opt['taxa']))
    tuning_df = tuning_results.get_dataframe(filter_metric = 'validation/test_score', filter_mode = 'max')
    tuning_df.index = [opt['taxa']] * len(tuning_df)
    tuning_df.to_csv(outfile)

    # delete sim files
    for file in glob(os.path.join(os.path.dirname(data['profile']),'*sim*')):
        if os.path.isdir(file):
            rmtree(file)
        else:
            os.remove(file)

# CLI argument
################################################################################
parser = argparse.ArgumentParser(description="This script executes tuning of a given model on a chosen taxa for Caribou's Sciki-learn models")

parser.add_argument('-d','--data', required=True, type=Path, help='Path to .npz data for extracted k-mers profile of bacteria')
parser.add_argument('-dh','--data_host', default=False, type=Path, help='Path to .npz data for extracted k-mers profile of host')
parser.add_argument('-dt','--database_name', required=True, help='Name of the bacteria database used to name files')
parser.add_argument('-ds','--host_name', default=False, help='Name of the host database used to name files')
parser.add_argument('-c','--classifier', required=True, choices=['onesvm','linearsvm','sgd','mnb'], help='Name of the classifier to tune')
parser.add_argument('-bs','--batch_size', required=True, help='Size of the batches to pass while training')
parser.add_argument('-t','--taxa', required=True, help='The taxa for which the tuning should be done')
parser.add_argument('-k','--kmers_length', required=True, help='Length of k-mers')
parser.add_argument('-o','--outdir', required=True, type=Path, help='Path to folder for outputing tuning results')
parser.add_argument('-wd','--workdir', default='/tmp/spill', type=Path, help='Optional. Path to a working directory where tuning data will be spilled')

args = parser.parse_args()

opt = vars(args)

ray.init(
    logging_level=logging.ERROR,
    _system_config={
        'object_spilling_config': json.dumps(
            {'type': 'filesystem', 'params': {'directory_path': str(opt['workdir'])}})
    }
)

# Data
################################################################################
data, datasets, preprocessor, labels_map = load_data(opt)

labels_list = [mapping[1] for mapping in labels_map]

# Model parameters
################################################################################
clf, train_params, tune_params = get_clf_params(opt['classifier'])

# Trainer
################################################################################

for name, ds in datasets.items():
    ds = ray.get(ds)
    ds = preprocessor.transform(ds)
    datasets[name] = ray.put(ds)

trainer = SklearnPartialTrainer(
    estimator = clf,
    label_column = opt['taxa'],
    labels_list = labels_list,
    features_list = data['kmers'],
    params = train_params,
    scoring = 'accuracy',
    datasets = datasets,
    batch_size = int(opt['batch_size']),
    set_estimator_cpus = True,
    scaling_config = ScalingConfig(
        trainer_resources = {
            'CPU' : 5
        }
    )
)

# Tuning parallelisation
################################################################################
tuner = Tuner(
    trainer,
    param_space = tune_params,
    tune_config = TuneConfig(
        search_alg=BasicVariantGenerator(
            max_concurrent = 8
        ),
        scheduler = ASHAScheduler(metric = 'test/test_score', mode = 'max')
    ),
    run_config = RunConfig(
        name = opt['classifier'],
        local_dir = opt['workdir']
    )
)
tuning_results = tuner.fit()

# Tuning results
################################################################################
output_results(opt, tuning_results)

print(f'Hyperparameters tuning of {opt["classifier"]} successfuly completed!')