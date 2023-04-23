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
from models.ray_tensor_min_max import TensorMinMaxScaler
from models.ray_tensor_max_abs import TensorMaxAbsScaler
from ray.data.preprocessors import Chain, BatchMapper, LabelEncoder
# Training
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDOneClassSVM, SGDClassifier
# Tuning
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig, ScalingConfig
from ray.tune.sklearn import TuneSearchCV, TuneGridSearchCV
from ray.tune.search.basic_variant import BasicVariantGenerator
# Predicting
from ray.train.sklearn import SklearnPredictor
from ray.train.batch_predictor import BatchPredictor
from joblib import Parallel, delayed, parallel_backend
# Parent class
from models.ray_utils import ModelsUtils
from models.sklearn.ray_sklearn_partial_trainer import SklearnPartialTrainer
from models.sklearn.ray_sklearn_probability_predictor import SklearnProbaPredictor

warnings.simplefilter(action='ignore')

# Functions
def merge_db_host(db_data, host_data):
    merged_database_host = {}
    merged_database_host['profile'] = "{}_host_merged".format(os.path.splitext(db_data["profile"])[0]) # Kmers profile

    df_classes = pd.DataFrame(db_data["classes"], columns=db_data["taxas"])
    df_cls_host = pd.DataFrame(host_data["classes"], columns=host_data["taxas"])
    if len(df_cls_host) > len(host_data['ids']):
        to_remove = np.arange(len(df_cls_host) - len(host_data['ids']))
        df_cls_host = df_cls_host.drop(to_remove, axis=0)
    elif len(df_cls_host) < len(host_data['ids']):
        diff = len(host_data['ids']) - len(df_cls_host)
        row = df_cls_host.iloc[0]
        for i in range(diff):
            df_cls_host = pd.concat([df_cls_host, row.to_frame().T], ignore_index=True)
    df_classes = pd.concat([df_classes, df_cls_host], ignore_index=True)
    for col in df_classes.columns:
        df_classes[col] = df_classes[col].str.lower()
    if len(np.unique(df_classes['domain'])) > 1:
        df_classes.loc[df_classes['domain'] == 'archaea','domain'] = 'bacteria'
    merged_database_host['classes'] = np.array(df_classes)  # Class labels
    merged_database_host['ids'] = np.concatenate((db_data["ids"], host_data["ids"]))  # IDs
    merged_database_host['kmers'] = db_data["kmers"]  # Features
    merged_database_host['taxas'] = db_data["taxas"]  # Known taxas for classification
    merged_database_host['fasta'] = (db_data['fasta'], host_data['fasta'])  # Fasta file needed for reads simulation
    
    df_db = ray.data.read_parquet(db_data["profile"])
    df_host = ray.data.read_parquet(host_data["profile"])
    df_merged = df_db.union(df_host)
    df_merged.write_parquet(merged_database_host['profile'])
    
    return merged_database_host

# CLI argument
################################################################################
parser = argparse.ArgumentParser(description="This script executes tuning of a given model on a chosen taxa for Caribou's Sciki-learn models")

parser.add_argument('-d','--data', required=True, type=Path, help='Path to .npz data for extracted k-mers profile of bacteria')
parser.add_argument('-dh','--data_host', default=None, type=Path, help='Path to .npz data for extracted k-mers profile of host')
parser.add_argument('-dt','--database_name', required=True, help='Name of the bacteria database used to name files')
parser.add_argument('-ds','--host_name', default=None, help='Name of the host database used to name files')
parser.add_argument('-c','--classifier', required=True, choices=['onesvm','linearsvm','sgd','mnb'], help='Name of the classifier to tune')
parser.add_argument('-t','--taxa', required=True, help='The taxa for which the tuning should be done')
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
db_data = verify_load_data(opt['data'])
if opt['host_name'] is not None:
    host_data = verify_load_data(opt['data_host'])
    db_data = merge_db_host(db_data, host_data)

db_df = ray.data.read_parquet(db_data['profile'])
db_cls = pd.read_csv(db_data['classes'], columns = db_data['taxas'])

for col in db_cls.columns:
    db_cls[col] = db_cls[col].str.lower()

if 'domain' in db_cls.columns:
    db_cls.loc[db_cls['domain'] != 'bacteria','domain'] = 'bacteria'

db_df = zip_X_y(db_df,db_cls)

# Preprocessing
################################################################################
col2drop = [col for col in db_df.schema().names if col not in ['__value__',opt['taxa']]]
db_df = db_df.drop_columns(col2drop)

if opt['classifier'] == 'onesvm':
    preprocessor = Chain(
        TensorMinMaxScaler(db_data['kmers']),
        OneClassSVMLabelEncoder('domain')
    )
    db_df = preprocessor.fit_transform(db_df)
    encoded = np.array([1,-1], dtype = np.int32)
    labels_map = zip(
        np.array(['bacteria', 'unknown'], dtype = object),
        encoded
    )
else:
    preprocessor = Chain(
        TensorMinMaxScaler(db_data['kmers']),
        LabelEncoder(['taxa'])
    )
    db_df = preprocessor.fit_transform(db_df)
    labels = list(preprocessor.preprocessors[1].stats_[f'unique_values({opt["taxa"]})'].keys())
    encoded = np.arange(len(labels))
    labels_map = zip(labels, encoded)

X_train = db_df.to_pandas()['__value__']
X_train = pd.DataFrame(X_train['__value__'].to_list()).to_numpy()
X_train = pd.DataFrame(X_train, columns = db_data['kmers'])
y_train = db_df.to_pandas()[opt['taxa']]
y_train = y_train.to_numpy()

# Model parameters
################################################################################
if opt['classifier'] == 'onesvm':
    clf = SGDOneClassSVM()
    tune_params = {
        'nu' : np.linspace(0.1, 1, 10),
        'learning_rate' : tune.grid_search(['constant','optimal','invscaling','adaptive']),
        'eta0' : np.logspace(-4,4,10),
        'tol' : [1e-3]
    }
elif opt['classifier'] == 'linearsvm':
    clf = SGDClassifier()
    tune_params = {
        'loss' : ['hinge'],
        'penalty' : tune.grid_search(['l2', 'l1', 'elasticnet']),
        'alpha' : np.logspace(-4,4,10),
        'learning_rate' : tune.grid_search(['constant','optimal','invscaling','adaptive']),
        'eta0' : np.logspace(-4,4,10)
    }
elif opt['classifier'] == 'sgd':
    clf = SGDClassifier()
    tune_params = {
        'params' : {
            'loss' : tune.grid_search(['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron']),
            'penalty' : tune.grid_search(['l2', 'l1', 'elasticnet']),
            'alpha' : np.logspace(-4,4,10),
            'learning_rate' : tune.grid_search(['constant','optimal','invscaling','adaptive']),
            'eta0' : np.logspace(-4,4,10)
        }
    }
elif opt['classifier'] == 'mnb':
    clf = MultinomialNB()
    tune_params = {
        'params' : {
            'alpha' : tune.grid_search(np.linspace(0,1,10)),
            'fit_prior' : tune.grid_search([True,False])
        }
    }

# Trainer/Tuner definition
################################################################################

tune_search = TuneGridSearchCV(
    clf,
    tune_params,
    scoring = 'accuracy',
    n_jobs = -1,
    early_stopping = True,
    max_iters = 100
)

# Tuning results
################################################################################

results = tune_search.fit(X_train, y_train)

results = pd.DataFrame(results.cv_results_)

results.to_csv(os.path.join(opt['outdir'],f'{opt["classifier"]}_{opt["taxa"]}_tuning_results.csv'), index = False)

print(f'Hyperparameters tuning of {opt["classifier"]} successfuly completed!')