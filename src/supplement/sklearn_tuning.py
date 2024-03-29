#!/usr/bin python3

import os
import ray
import warnings
import argparse
import numpy as np
import pandas as pd

from glob import glob
from pathlib import Path

# Package modules
from utils import *
from models.reads_simulation import readsSimulation
from models.ray_tensor_min_max import TensorMinMaxScaler

# from ray.data.preprocessors import MinMaxScaler
from src.models.sklearn.partial_trainer import SklearnPartialTrainer
from src.models.encoders.onesvm_label_encoder import OneClassSVMLabelEncoder

# Preprocessing
from ray.data.preprocessors import Chain, LabelEncoder

# Training
from models.sklearn.scoring_one_svm import ScoringSGDOneClassSVM
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

# Tuning
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig, ScalingConfig


warnings.simplefilter(action='ignore')

# Functions
################################################################################

def merge_db_host(db_data, host_data):
    merged_db_host = {}
    merged_db_host['profile'] = f"{db_data['profile']}_host_merged" # Kmers profile

    if os.path.exists(merged_db_host['profile']):
        files_lst = glob(os.path.join(merged_db_host['profile'], '*.parquet'))
        merged_ds = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))
    else:
        files_lst = glob(os.path.join(db_data['profile'], '*.parquet'))
        db_ds = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))
        files_lst = glob(os.path.join(host_data['profile'], '*.parquet'))
        host_ds = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))

        col2drop = []
        for col in db_ds.schema().names:
            if col not in ['id','domain','__value__']:
                col2drop.append(col)
        db_ds = db_ds.drop_columns(col2drop)
        col2drop = []
        for col in host_ds.schema().names:
            if col not in ['id','domain','__value__']:
                col2drop.append(col)
        host_ds = host_ds.drop_columns(col2drop)

        merged_ds = db_ds.union(host_ds)
        merged_ds.write_parquet(merged_db_host['profile'])
    
    merged_db_host['ids'] = np.concatenate((db_data["ids"], host_data["ids"]))  # IDs
    merged_db_host['kmers'] = db_data['kmers']  # Features
    merged_db_host['taxas'] = ['domain']  # Known taxas for classification
    merged_db_host['fasta'] = (db_data['fasta'], host_data['fasta'])  # Fasta file needed for reads simulation
    
    return merged_db_host, merged_ds

def sim_4_cv(ds, database_data, name):
    print('_sim_4_cv')
    k = len(database_data['kmers'][0])
    cols = ['id']
    cols.extend(database_data['taxas'])
    cls = pd.DataFrame(columns = cols)
    for batch in ds.iter_batches(batch_format = 'pandas'):
        cls = pd.concat([cls, batch[cols]], axis = 0, ignore_index = True)
    
    sim_outdir = os.path.dirname(database_data['profile'])
    print(f'nb samples : {len(list(cls["id"]))}')
    cv_sim = readsSimulation(database_data['fasta'], cls, list(cls['id']), 'miseq', sim_outdir, name)
    sim_data = cv_sim.simulation(k, database_data['kmers'])
    files_lst = glob(os.path.join(sim_data['profile'], '*.parquet'))
    ds = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))
    return ds

def convert_archaea_bacteria(df):
    df.loc[df['domain'].str.lower() == 'archaea', 'domain'] = 'Bacteria'
    return df

def verify_load_host_merge(db_data, host_data):
    db_data = verify_load_data(db_data)
    host_data = verify_load_data(host_data)
    merged_data, merged_ds = merge_db_host(db_data, host_data)
    merged_ds = merged_ds.map_batches(
        convert_archaea_bacteria,
        batch_format = 'pandas'
    )
    return merged_data, merged_ds

def split_val_test_ds(ds, data):
    val_path = os.path.join(os.path.dirname(data['profile']), f'Xy_genome_simulation_validation_data_K{len(data["kmers"][0])}')
    test_path = os.path.join(os.path.dirname(data['profile']), f'Xy_genome_simulation_test_data_K{len(data["kmers"][0])}')
    if os.path.exists(val_path):
        files_lst = glob(os.path.join(val_path, '*.parquet'))
        val_ds = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))
        val_ds = val_ds.map_batches(
            convert_archaea_bacteria,
            batch_format = 'pandas'
        )
    else:
        val_ds = ds.random_sample(0.1)
        if val_ds.count() == 0:
            nb_smpl = round(ds.count() * 0.1)
            val_ds = ds.random_shuffle().limit(nb_smpl)
        val_ds = sim_4_cv(val_ds, data, 'validation')
    if os.path.exists(test_path):
        files_lst = glob(os.path.join(test_path, '*.parquet'))
        test_ds = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))
        test_ds = test_ds.map_batches(
            convert_archaea_bacteria,
            batch_format = 'pandas'
        )
    else:
        test_ds = ds.random_sample(0.1)
        if test_ds.count() == 0:
            nb_smpl = round(ds.count() * 0.1)
            test_ds = ds.random_shuffle().limit(nb_smpl)
        test_ds = sim_4_cv(test_ds, data, 'test')
    return val_ds, test_ds

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

init_ray_cluster(opt['workdir'])

# Data
################################################################################
print('data loading')

if opt['classifier'] == 'onesvm' and opt['taxa'] == 'domain':
    if opt['data_host'] is None:
        raise ValueError('To tune for a domain taxa, a host species is required.\
                        It is used to confirm that the models can discern other sequences than bacteria.')
    else:
        test_val_data, test_val_ds = verify_load_host_merge(opt['data'], opt['data_host'])
        val_ds, test_ds = split_val_test_ds(test_val_ds,test_val_data)
        db_data = verify_load_data(opt['data'])
        files_lst = glob(os.path.join(db_data['profile'], '*.parquet'))
        train_ds = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))
elif opt['classifier'] == 'linearsvm' and opt['taxa'] == 'domain':
    if opt['data_host'] is None:
        raise ValueError('To tune for a domain taxa, a host species is required.\
                        It is used to confirm that the models can discern other sequences than bacteria.')
    else:
        db_data, train_ds = verify_load_host_merge(opt['data'], opt['data_host'])
        val_ds, test_ds = split_val_test_ds(train_ds, db_data)
else:
    db_data = verify_load_data(opt['data'])
    files_lst = glob(os.path.join(db_data['profile'], '*.parquet'))
    train_ds = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))
    val_ds, test_ds = split_val_test_ds(train_ds, db_data)

# Preprocessing
################################################################################
print('data preprocessing')

if opt['classifier'] == 'onesvm':
    preprocessor = Chain(
        TensorMinMaxScaler(db_data['kmers']),
        OneClassSVMLabelEncoder('domain')
    )
    train_ds = train_ds.map_batches(
        convert_archaea_bacteria,
        batch_format = 'pandas'
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

datasets = {
    'train' : ray.put(train_ds.materialize()),
    'test' : ray.put(test_ds.materialize()),
    'validation' : ray.put(val_ds.materialize())
}

# Model parameters
################################################################################
print('model params')

if opt['classifier'] == 'onesvm':
    clf = ScoringSGDOneClassSVM()
    num_samples = 20
    train_params = {
        'nu' : 0.1,
        'learning_rate' : 'constant',
        'tol' : 1e-3,
        'eta0' : 0.001
    }
    tune_params = {
        'params' : {
            'nu' : tune.uniform(0,1),
            'learning_rate' : tune.grid_search(['constant','optimal','invscaling','adaptive']),
        }
    }
elif opt['classifier'] == 'linearsvm':
    clf = SGDClassifier()
    # num_samples = 60
    train_params = {
        'loss' : 'hinge',
        'penalty' : 'l2',
        'alpha' : 4,
        'learning_rate' : 'constant',
        'eta0' : 0.001,
        'n_jobs' : -1
    }
    tune_params = {
        'params' : {
            'penalty' : tune.grid_search(['l2', 'l1', 'elasticnet']),
            'alpha' : tune.loguniform(1e-4,1e4),
            'learning_rate' : tune.grid_search(['constant','optimal','invscaling','adaptive']),
        }
    }
elif opt['classifier'] == 'sgd':
    clf = SGDClassifier()
    # num_samples = 300
    train_params = {
        'loss' : 'hinge',
        'penalty' : 'l2',
        'alpha' : 4,
        'learning_rate' : 'constant',
        'eta0' : 0.001,
        'n_jobs' : -1
    }
    tune_params = {
        'params' : {
            'loss' : tune.grid_search(['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron']),
            'penalty' : tune.grid_search(['l2', 'l1', 'elasticnet']),
            'alpha' : tune.loguniform(1e-4,1e4),
            'learning_rate' : tune.grid_search(['constant','optimal','invscaling','adaptive']),
        }
    }
elif opt['classifier'] == 'mnb':
    clf = MultinomialNB()
    # num_samples = 10
    train_params = {
        'alpha' : 1.0e-10,
        'fit_prior' : True
    }
    tune_params = {
        'params' : {
            'alpha' : tune.uniform(0,1),
            'fit_prior' : tune.grid_search([True,False])
        }
    }

# Trainer/Tuner definition
################################################################################
# Basic Ray tuner using GridSearch algo
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
    run_config=RunConfig(
        name = opt['classifier'],
        local_dir = opt['workdir']
    ),
)
print('tuner')
tuner = Tuner(
    trainer,
    param_space = tune_params,
    tune_config = TuneConfig(
        num_samples = 5,
        max_concurrent_trials = int((os.cpu_count() * 0.8)),
        scheduler = ASHAScheduler(
            metric = 'test/test_score', # mean accuracy according to scikit-learn's doc
            mode = 'max'
        )
    )
)

# Tuning results
################################################################################

results = tuner.fit()

results = results.get_dataframe()

results.to_csv(os.path.join(opt['outdir'],f'{opt["classifier"]}_{opt["taxa"]}_tuning_results.csv'), index = False)

print(f'Hyperparameters tuning of {opt["classifier"]} successfuly completed!')