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
from joblib import Parallel, delayed, parallel_backend

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

ray.init()

# Functions
################################################################################

# Function from class function models.ray_sklearn.SklearnModel._training_preprocess
def preprocess(X, y, cols, taxa):
    df = X.add_column([taxa, 'id'], lambda x : y)
    preprocessor = Chain(
        SimpleImputer(
            cols,
            strategy = 'constant',
            fill_value = 0
        ),
        MinMaxScaler(cols)
    )
    df = preprocessor.fit_transform(df)
    labels = np.unique(y[taxa])
    encoder = LabelEncoder(taxa)
    df = encoder.fit_transform(df)
    return df

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
        ids = []
        for row in df.iter_rows():
            ids.append(row['__index_level_0__'])
        labels = pd.DataFrame(sim_data['classes'], index = ids)
        df = df.add_column(taxa, lambda x : labels)
        return df

# CLI argument
################################################################################
parser = argparse.ArgumentParser(description='This script extracts K-mers of the given dataset using the available ressources on the computer before saving it to drive.')

parser.add_argument('-d','--data', required=True, type=Path, help='Path to .npz data for extracted k-mers profile of bacteria')
parser.add_argument('-dh','--data_host', default=False, type=Path, help='Path to .npz data for extracted k-mers profile of host')
parser.add_argument('-dt','--database_name', required=True, help='Name of the bacteria database used to name files')
parser.add_argument('-ds','--host_name', default=False, help='Name of the host database used to name files')
parser.add_argument('-c','--classifier', required=True, choices=['onesvm','linearsvm','sgd','mnb'], help='Name of the classifier to tune')
parser.add_argument('-bs','--batch_size', required=True, help='Size of the batches to pass while training')
parser.add_argument('-k','--kmers_length', required=True, help='Length of k-mers')
parser.add_argument('-o','--out', required=True, type=Path, help='Path to outdir')

args = parser.parse_args()

opt = vars(args)

# Data
################################################################################
data = None

if opt['data_host']:
    data_db = load_Xy_data(opt['data'])
    data_host = load_Xy_data(opt['data_host'])
    data = merge_database_host(data_db, data_host)
else:
    data = load_Xy_data(opt['data'])

taxas = data['taxas'].copy()

if opt['classifier'] in ['onesvm', 'linearsvm']:
    taxas = [taxas[-1]]
    result_file = os.path.join(opt['out'],'tuning_{}_{}_classifier_{}_k{}.csv'.format(opt['database_name'],opt['host_name'],opt['classifier'],opt['kmers_length']))
else:
    del taxas[-1]
    result_file = os.path.join(opt['out'],'tuning_{}_classifier_{}_k{}.csv'.format(opt['database_name'],opt['classifier'],opt['kmers_length']))

tuning_df = pd.DataFrame(np.empty((len(taxas), 3)), columns = ['precision','recall','f1_weighted'], index = taxas)

for taxa in taxas:
    X = ray.data.read_parquet(data['profile'])
    cols = list(X.limit(1).to_pandas().columns)
    ids = list(X.to_pandas().index)
    y = pd.DataFrame(
        {taxa:pd.DataFrame(data['classes'], columns = data['taxas']).loc[:,taxa].astype('string').str.lower(),
        'id' : ids}
    )
    y.index = ids
    labels_list = np.unique(y[taxa])
    df = preprocess(X, y, cols, taxa)

    df_train, df_val = df.train_test_split(0.2, shuffle = True)
    df_train = df_train.drop_columns(['id'])
    df_val = sim_4_cv(df_val, data, 'tuning_val', taxa, cols, opt['kmers_length'])

    datasets = {'train' : ray.put(df_train), 'validation' : ray.put(df_val)}

# Model parameters
################################################################################
    clf = None
    train_params = {}
    tune_params = []

    if opt['classifier'] == 'onesvm':
        clf = SGDOneClassSVM()
        train_params = {
            'nu' : 0.05,
            'tol' : 1e-4
        }

        tune_params = []
        for  lr in ['constant','optimal','invscaling','adaptive']:
            tune_params.append({
                'params' : {
                    'nu' : tune.grid_search(np.logspace(-4,4,10)),
                    'learning_rate' : lr,
                    'eta0' : tune.grid_search(np.logspace(-4,4,10))
                }
            })

    elif opt['classifier'] == 'linearsvm' or opt['classifier'] == 'sgd':
        clf = SGDClassifier()
        train_params = {
            'loss' : 'squared_error'
        }
        tune_params = []
        for loss in ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']:
            for penalty in ['l2', 'l1', 'elasticnet']:
                for lr in ['constant','optimal','invscaling','adaptive']:
                    tune_params.append({
                        'params' : {
                            'loss' : loss,
                            'penalty' : penalty,
                            'alpha' : tune.grid_search(np.logspace(-4,4,10)),
                            'learning_rate' : lr,
                            'eta0' : tune.grid_search(np.logspace(-4,4,10))
                        }
                    })

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
        label_column = taxa,
        labels_list = labels_list,
        params = train_params,
        scoring = ['precision','recall','f1_weighted'],
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
            metric = 'validation/test_f1_weighted',
            mode = 'max',
            scheduler = ASHAScheduler()
        ),
        run_config = RunConfig(
            name = opt['classifier']
        )
    )
    tuning_results = tuner.fit()

# Tuning results
################################################################################

    best_metric = tuning_results.get_best_result().metric
    tuning_df.loc[taxa,'precision'] = best_metric['validation']['test_precision']
    tuning_df.loc[taxa,'recall'] = best_metric['validation']['test_recall']
    tuning_df.loc[taxa,'f1_weighted'] = best_metric['validation']['test_f1_weighted']

    # delete sim files
    for file in glob(os.path.join(os.path.dirname(kmers_ds['profile']),'*sim*')):
        if os.path.isdir(file):
            rmtree(file)
        else:
            os.remove(file)

tuning_df.to_csv(result_file)
