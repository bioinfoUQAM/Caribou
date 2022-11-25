#!/usr/bin python3

import os
import ray
import argparse
import warnings
import numpy as np
import pandas as pd

from glob import glob
from pathlib import Path
from shutil import rmtree

# Package modules
from utils import *
from models.reads_simulation import readsSimulation
from models.ray_sklearn_partial_trainer import SklearnPartialTrainer

# Preprocessing
from ray.data.preprocessors import MinMaxScaler, LabelEncoder, Chain, SimpleImputer

# Training
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDOneClassSVM, SGDClassifier

# Tuning
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig, ScalingConfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Functions
################################################################################

# Function from class function models.classification.ClassificationMethods._merge_database_host
def merge_database_host(database_data, host_data):
    merged_database_host = {}

    merged_database_host['profile'] = "{}_host_merged".format(os.path.splitext(database_data["profile"])[0]) # Kmers profile

    df_classes = pd.DataFrame(database_data["classes"], columns=database_data["taxas"])
    if len(np.unique(df_classes['domain'])) != 1:
        df_classes[df_classes['domain'] != 'bacteria'] = 'bacteria'
    df_classes = df_classes.append(pd.DataFrame(host_data["classes"], columns=host_data["taxas"]), ignore_index=True)
    merged_database_host['classes'] = np.array(df_classes)  # Class labels
    merged_database_host['kmers'] = database_data["kmers"]  # Features
    merged_database_host['taxas'] = database_data["taxas"]  # Known taxas for classification
    merged_database_host['fasta'] = (database_data['fasta'], host_data['fasta'])  # Fasta file needed for reads simulation

    df_db = ray.data.read_parquet(database_data["profile"])
    df_host = ray.data.read_parquet(host_data["profile"])
    df_merged = df_db.union(df_host)
    df_merged.write_parquet(merged_database_host['profile'])
    return merged_database_host

# Function from class function models.ray_sklearn.SklearnModel._training_preprocess
def preprocess(X, y, cols, taxa):
    print:('X : ',X)
    print('y : ',y)
    print('taxa : ',taxa)
    raise ValueError
    df = X.add_column([taxa], lambda x : y)
    print(df.take(1).show())
    df = preprocess_values(df, cols)
    df, labels = preprocess_labels(df, taxa)
   
    return (df, labels)

def preprocess_values(df, cols):
    cols_batches = np.array_split(cols, 1000)
    for batch in cols_batches:
        min_pos = cols.index(batch[0])
        max_pos = cols.index(batch[-1])
        for col in batch:
            df = df.add_column(col, lambda df: df['__value__'].to_numpy()[0][cols.index(col)])
        preprocessor = Chain(
            SimpleImputer(
                batch,
                strategy='constant',
                fill_value=0
            ),
            MinMaxScaler(batch)
        )
        df = preprocessor.fit_transform(df)
        for row in df.iter_rows():
            row['__value__'][0][min_pos:max_pos+1] = np.array(row[batch])
            print(row)
        df = df.drop_columns(batch)
    MinMaxScaler(cols)

def preprocess_labels(df, taxa):
    labels = np.unique(y[taxa])
    encoder = LabelEncoder(taxa)
    df = encoder.fit_transform(df)
    return df, labels

# Function from class models.ray_utils.ModelsUtils
def sim_4_cv(df, kmers_ds, name, taxa, cols, k):
        sim_genomes = []
        for row in df.iter_rows():
            sim_genomes.append(row['id'])
        cls = pd.DataFrame({'id':sim_genomes,taxa:df.to_pandas()[taxa]})
        sim_outdir = os.path.dirname(kmers_ds['profile'])
        cv_sim = readsSimulation(kmers_ds['fasta'], cls, sim_genomes, 'miseq', sim_outdir, name)
        sim_data = cv_sim.simulation(k, cols)
        df = ray.data.read_parquet(sim_data['profile'])
        for row in df.iter_rows():
            ids.append(row['__index_level_0__'])
        labels = pd.DataFrame(sim_data['classes'], index = kmers_ds['ids'])
        df = df.add_column(taxa, lambda x : labels)
        return df

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
parser.add_argument('-o','--outfile', required=True, type=Path, help='Path to outfile')
parser.add_argument('-wd','--workdir', default='~/ray_results', type=Path, help='Optional. Path to a working directory where Ray Tune will output and spill tuning data')

args = parser.parse_args()

opt = vars(args)

ray.init()

# Data
################################################################################
data = None

if opt['data_host']:
    data_db = load_Xy_data(opt['data'])
    data_host = load_Xy_data(opt['data_host'])
    data = merge_database_host(data_db, data_host)
else:
    data = load_Xy_data(opt['data'])

X = ray.data.read_parquet(data['profile'])
cols = data['kmers']
y = pd.DataFrame(
    {opt['taxa']:pd.DataFrame(data['classes'], columns = data['taxas']).loc[:,opt['taxa']].astype('string').str.lower(),
    'id' : data['ids']}
)
df_label = preprocess(X, y, cols, opt['taxa'])
df = df_label[0]
labels = df_label[1]

df_train, df_val = df.train_test_split(0.2, shuffle = True)
df_train = df_train.drop_columns(['id'])
df_val = sim_4_cv(df_val, data, 'tuning_val', opt['taxa'], cols, opt['kmers_length'])

datasets = {'train' : ray.put(df_train), 'validation' : ray.put(df_val)}

# Model parameters
################################################################################
clf = None
train_params = {}
tune_params = {}

if opt['classifier'] == 'onesvm':
    clf = SGDOneClassSVM()
    train_params = {
        'nu' : 0.05,
        'tol' : 1e-4
    }
    tune_params = {
        'params' : {
            'nu' : tune.grid_search(np.logspace(-4,4,10)),
            'learning_rate' : tune.grid_search(['constant','optimal','invscaling','adaptive']),
            'eta0' : tune.grid_search(np.logspace(-4,4,10))
        }
    }
elif opt['classifier'] == 'linearsvm' or opt['classifier'] == 'sgd':
    clf = SGDClassifier()
    train_params = {
        'loss' : 'squared_error'
    }
    tune_params = {
        'params' : {
            'loss' : tune.grid_search(['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
            'penalty' : tune.grid_search(['l2', 'l1', 'elasticnet']),
            'alpha' : tune.grid_search(np.logspace(-4,4,10)),
            'learning_rate' : tune.grid_search(['constant','optimal','invscaling','adaptive']),
            'eta0' : tune.grid_search(np.logspace(-4,4,10))
        }
    }
elif opt['classifier'] == 'mnb':
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

# Trainer
################################################################################
trainer = SklearnPartialTrainer(
    estimator = clf,
    label_column = opt['taxa'],
    labels_list = labels_list,
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
        metric = 'validation/test_score',
        mode = 'max',
        scheduler = ASHAScheduler()
    ),
    run_config = RunConfig(
        name = opt['classifier'],
        local_dir = opt['workdir']
    )
)
tuning_results = tuner.fit()

# Tuning results
################################################################################

tuning_df = tuning_results.get_dataframe(filter_metric = 'validation/test_score', filter_mode = 'max')
tuning_df.index = [opt['taxa']] * len(tuning_df)
tuning_df.to_csv(opt['outfile'])

# delete sim files
for file in glob(os.path.join(os.path.dirname(data['profile']),'*sim*')):
    if os.path.isdir(file):
        rmtree(file)
    else:
        os.remove(file)
