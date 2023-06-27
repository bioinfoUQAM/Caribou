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

from pathlib import Path

# Package modules
from utils import *
from models.reads_simulation import readsSimulation
from models.ray_tensor_min_max import TensorMinMaxScaler
from models.sklearn.ray_sklearn_partial_trainer import SklearnPartialTrainer
from models.sklearn.ray_sklearn_onesvm_encoder import OneClassSVMLabelEncoder

# Preprocessing
from ray.data.preprocessors import Chain, BatchMapper, LabelEncoder
# Training
from supplement.scoring_one_svm import ScoringSGDOneClassSVM
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDOneClassSVM, SGDClassifier
# Tuning
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig, ScalingConfig
# Parent class
from models.sklearn.ray_sklearn_partial_trainer import SklearnPartialTrainer

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

def sim_4_cv(df, database_data, name):
    print('_sim_4_cv')
    k = len(database_data['kmers'][0])
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
    sim_outdir = os.path.dirname(database_data['profile'])
    cv_sim = readsSimulation(database_data['fasta'], cls, sim_cls_dct['id'], 'miseq', sim_outdir, name)
    sim_data = cv_sim.simulation(k, database_data['kmers'])
    sim_cls = pd.DataFrame(sim_data['classes'], columns = sim_data['taxas'])
    df = ray.data.read_parquet(sim_data['profile'])
    df = zip_X_y(df, sim_cls)
    return df

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
print('data loading')

db_data = verify_load_data(opt['data'])
if opt['host_name'] is not None and opt['taxa'] == 'domain':
    host_data = verify_load_data(opt['data_host'])
    db_data = merge_db_host(db_data, host_data)

train_ds = ray.data.read_parquet(db_data['profile'])
db_cls = pd.DataFrame(db_data['classes'], columns = db_data['taxas'])

for col in db_cls.columns:
    db_cls[col] = db_cls[col].str.lower()

if 'domain' in db_cls.columns:
    db_cls.loc[db_cls['domain'] == 'archaea','domain'] = 'bacteria'

train_ds = zip_X_y(train_ds,db_cls)

# Preprocessing
################################################################################
print('data preprocessing')

val_ds = train_ds.random_sample(0.1)
test_ds = train_ds.random_sample(0.1)
if val_ds.count() == 0:
    nb_smpl = round(val_ds.count() * 0.1)
    val_ds = val_ds.limit(nb_smpl)
if test_ds.count() == 0:
    nb_smpl = round(test_ds.count() * 0.1)
    test_ds = test_ds.limit(nb_smpl)
val_ds = sim_4_cv(train_ds, db_data, 'validation')
test_ds = sim_4_cv(train_ds, db_data, 'test')

col2drop = [col for col in train_ds.schema().names if col not in ['__value__',opt['taxa']]]
train_ds = train_ds.drop_columns(col2drop)
val_ds = val_ds.drop_columns(col2drop)
test_ds = test_ds.drop_columns(col2drop)

if opt['classifier'] == 'onesvm':
    preprocessor = Chain(
        TensorMinMaxScaler(db_data['kmers']),
        OneClassSVMLabelEncoder('domain')
    )
    train_ds = preprocessor.fit_transform(train_ds)
    val_ds = preprocessor.transform(val_ds)
    test_ds = preprocessor.transform(test_ds)
    encoded = np.array([1,-1], dtype = np.int32)
    labels_map = zip(
        np.array(['bacteria', 'unknown'], dtype = object),
        encoded
    )
else:
    preprocessor = Chain(
        TensorMinMaxScaler(db_data['kmers']),
        LabelEncoder(opt['taxa'])
    )
    train_ds = preprocessor.fit_transform(train_ds)
    val_ds = preprocessor.transform(val_ds)
    test_ds = preprocessor.transform(test_ds)
    labels = list(preprocessor.preprocessors[1].stats_[f'unique_values({opt["taxa"]})'].keys())
    encoded = np.arange(len(labels))
    labels_map = zip(labels, encoded)


"""
X_train = train_ds.to_pandas()['__value__']
X_train = pd.DataFrame(X_train.to_list()).to_numpy()
X_train = pd.DataFrame(X_train, columns = db_data['kmers'])
y_train = train_ds.to_pandas()[opt['taxa']]
y_train = y_train.to_numpy()
"""

datasets = {
    'train' : ray.put(train_ds),
    'test' : ray.put(test_ds),
    'validation' : ray.put(val_ds)
}

# Model parameters
################################################################################
print('model params')

if opt['classifier'] == 'onesvm':
    clf = ScoringSGDOneClassSVM()
    train_params = {
        'nu' : 0.1,
        'learning_rate' : 'constant',
        'eta0' : 4,
        'tol' : 1e-3
    }
    tune_params = {
        'params' : {
            'nu' : tune.grid_search(np.linspace(0.1, 1, 10)),
            'learning_rate' : tune.grid_search(['constant','optimal','invscaling','adaptive']),
            'eta0' : tune.grid_search(np.logspace(-4,4,10)),
            'tol' : tune.grid_search([1e-3])
        }
    }
elif opt['classifier'] == 'linearsvm':
    clf = SGDClassifier()
    train_params = {
        'loss' : 'hinge',
        'penalty' : 'l2',
        'alpha' : 4,
        'learning_rate' : 'constant',
        'eta0' : 4
    }
    tune_params = {
        'params' : {
            'loss' : tune.grid_search(['hinge']),
            'penalty' : tune.grid_search(['l2', 'l1', 'elasticnet']),
            'alpha' : tune.grid_search(np.logspace(0,5,10)),
            'learning_rate' : tune.grid_search(['constant','optimal','invscaling','adaptive']),
            'eta0' : tune.grid_search(np.logspace(-4,4,10))
        }
    }
elif opt['classifier'] == 'sgd':
    clf = SGDClassifier()
    train_params = {
        'loss' : 'hinge',
        'penalty' : 'l2',
        'alpha' : 4,
        'learning_rate' : 'constant',
        'eta0' : 4
    }
    tune_params = {
        'params' : {
            'loss' : tune.grid_search(['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron']),
            'penalty' : tune.grid_search(['l2', 'l1', 'elasticnet']),
            'alpha' : tune.grid_search(np.logspace(0,5,10)),
            'learning_rate' : tune.grid_search(['constant','optimal','invscaling','adaptive']),
            'eta0' : tune.grid_search(np.logspace(-4,4,10))
        }
    }
elif opt['classifier'] == 'mnb':
    clf = MultinomialNB()
    train_params = {
        'alpha' : 1.0e-10,
        'fit_prior' : True
    }
    tune_params = {
        'params' : {
            'alpha' : tune.grid_search(np.linspace(1.0e-10,1,10)),
            'fit_prior' : tune.grid_search([True,False])
        }
    }

# Trainer/Tuner definition
################################################################################
print('trainer')
trainer = SklearnPartialTrainer(
    estimator=clf,
    label_column=opt['taxa'],
    labels_list=encoded,
    features_list=db_data['kmers'],
    params=train_params,
    datasets=datasets,
    batch_size=4,
    training_epochs=10,
    set_estimator_cpus=True,
    scaling_config=ScalingConfig(
        trainer_resources={
            'CPU': int(os.cpu_count()*0.8)
        }
    ),
    run_config=RunConfig(
        name=opt['classifier'],
        local_dir=opt['workdir']
    ),
)

print('tuner')
tuner = Tuner(
    trainer,
    param_space=tune_params,
    tune_config=TuneConfig(
        max_concurrent_trials=5,
        scheduler=ASHAScheduler(
            metric = 'test/test_score', # mean accuracy according to scikit-learn's doc
            mode='max'
        )
    )
)



"""
print('tuning')
tune_search = TuneGridSearchCV(
    clf,
    tune_params,
    scoring = 'accuracy',
    n_jobs = -1,
    early_stopping = True,
    max_iters = 100
)
"""
# Tuning results
################################################################################
print('results output')

results = tuner.fit()

results = results.get_dataframe()

results.to_csv(os.path.join(opt['outdir'],f'{opt["classifier"]}_{opt["taxa"]}_tuning_results.csv'), index = False)

print(f'Hyperparameters tuning of {opt["classifier"]} successfuly completed!')