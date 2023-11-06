import os
import ray
import json
import logging
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa

from glob import glob
from pathlib import Path
from warnings import warn
from psutil import virtual_memory

__author__ = "Nicolas de Montigny"

__all__ = [
    'init_ray_cluster',
    'load_Xy_data',
    'save_Xy_data',
    'verify_file',
    'verify_fasta',
    'verify_data_path',
    'verify_saving_path',
    'verify_host',
    'verify_host_params',
    'verify_boolean',
    'verify_positive_int',
    'verify_0_1',
    'verify_binary_classifier',
    'verify_multiclass_classifier',
    'define_create_outdirs',
    'verify_seqfiles',
    'verify_kmers_list_length',
    'verify_load_data',
    'verify_concordance_klength',
    'verify_taxas',
    'verify_load_preclassified',
    'merge_save_data',
    'zip_X_y',
    'ensure_length_ds',
    'convert_archaea_bacteria',
    'verify_load_db',
    'verify_load_host_merge',
    'merge_db_host'
]

# Constants
#########################################################################################################

TENSOR_COLUMN_NAME = '__value__'

# System
#########################################################################################################

# Initialize ray cluster
def init_ray_cluster(workdir):
    mem = virtual_memory().total
    frac = 0.8
    while not ray.is_initialized():
        try:
            ray.init(
                object_store_memory = mem * frac,
                _temp_dir = str(workdir),
            )
            logging.getLogger("ray").setLevel(logging.WARNING)
            ray.data.DataContext.get_current().execution_options.verbose_progress = True
        except ValueError :
            ray.shutdown()
            frac -= 0.05

# Data I/O
#########################################################################################################

# Load data from file
def load_Xy_data(Xy_file):
    with np.load(Xy_file, allow_pickle=True) as f:
        return f['data'].tolist()

# Save data to file
def save_Xy_data(df, Xy_file):
    np.savez(Xy_file, data = df)

# User arguments verification
#########################################################################################################

def verify_file(file : Path):
    if file is not None and not os.path.exists(file):
        raise ValueError(f'Cannot find file {file} !')

def verify_fasta(file : Path):
    if not os.path.isfile(file) and not os.path.isdir(file):
        raise ValueError('Fasta must be an interleaved fasta file or a directory containing fasta files.')

def verify_data_path(dir : Path):
    if not os.path.exists(dir):
        raise ValueError(f"Cannot find data folder {dir} ! Exiting")

def verify_saving_path(dir : Path):
    path, folder = os.path.split(dir)
    if not os.path.exists(path):
        raise ValueError("Cannot find where to create output folder !")

def verify_host(host : str):
    if host not in ['none', 'None', None]:
        return host
    else:
        return None

def verify_host_params(host : str, host_seq_file : Path, host_cls_file : Path):
    host = verify_host(host)
    if host is not None:
        if host_seq_file is None or host_cls_file is None:
            raise ValueError(f'Please provide host sequence and classification files or remove the host name from config file!\n\
                actual values : host = {host}, host_seq_file = {host_seq_file}, host_cls_file = {host_cls_file}')

def verify_boolean(val : bool, parameter : str):
    if val not in [True,False,None]:
        raise ValueError(
            f'Invalid value for {parameter} ! Please use boolean values !\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')

def verify_positive_int(val : int, parameter : str):
    if type(val) != int or val < 0:
        raise ValueError(
            f'Invalid value for {parameter} ! Please use a positive integer !\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')

def verify_0_1(val : float, parameter : str):
    if type(val) != float:
        raise ValueError(
            f'Invalid value for {parameter} ! Please use a float between 0 and 1 !\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')
    elif not 0 <= val <= 1:
        raise ValueError(
            f'Invalid value for {parameter} ! Please use a float between 0 and 1 !\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')

def verify_binary_classifier(clf : str):
    if clf not in ['onesvm', 'linearsvm', 'attention', 'lstm', 'deeplstm']:
        raise ValueError(
            'Invalid host extraction classifier !\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')

def verify_multiclass_classifier(clf : str):
    if clf not in ['sgd', 'mnb', 'lstm_attention', 'cnn', 'widecnn']:
        raise ValueError(
            'Invalid multiclass bacterial classifier !\n' +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')

def verify_seqfiles(seqfile : Path, seqfile_host : Path):
    if seqfile is None and seqfile_host is None:
        raise ValueError("No fasta file to extract K-mers from !")
    else:
        if seqfile is not None:
            verify_fasta(seqfile)
        if seqfile_host is not None:
            verify_fasta(seqfile_host)

def verify_concordance_klength(klen1 : int, klen2 : int):
    if klen1 != klen2:
        raise ValueError("K length between datasets is inconsistent ! Exiting\n" +
                f"K length of bacteria dataset is {klen1} while K length from host is {klen2}")

# Verif + handling
#########################################################################################################

def define_create_outdirs(dir : Path):
    outdirs = {}
    verify_saving_path(dir)
    outdirs['main_outdir'] = Path(dir)
    outdirs['data_dir'] = Path(os.path.join(dir, 'data'))
    outdirs['models_dir'] = Path(os.path.join(dir, 'models'))
    outdirs['results_dir'] = Path(os.path.join(dir, 'results'))
    os.makedirs(dir, mode=0o700, exist_ok=True)
    os.makedirs(outdirs['data_dir'], mode=0o700, exist_ok=True)
    os.makedirs(outdirs['models_dir'], mode=0o700, exist_ok=True)
    os.makedirs(outdirs['results_dir'], mode=0o700, exist_ok=True)
    return outdirs

def verify_kmers_list_length(klen: int, kmers_file: Path):
    if kmers_file is not None:
        # Read kmers file to put in list
        kmers_list = []
        with open(kmers_file, 'r') as handle:
            kmers_list = [kmer.rstrip() for kmer in handle.readlines()]
        if klen <= 0:
            warn("Invalid K-mers length but K-mers list file was found ! Setting K-mers length to correspond to previously extracted length !")
            klen = len(kmers_list[0])
        elif klen != len(kmers_list[0]):
            warn("K-mers length is different than length in the K-mers list file given ! Setting K-mers length to correspond to previously extracted length !")
            klen = len(kmers_list[0])
        return klen, kmers_list
    else:
        verify_positive_int(klen, 'K-mers length')
        return klen, None

def verify_load_data(data_file: Path):
    verify_file(data_file)
    data = load_Xy_data(data_file)
    verify_data_path(data['profile'])
    if not isinstance(data['ids'], list):
        raise ValueError("Invalid data file !")
    elif not isinstance(data['kmers'], list):
        raise ValueError("Invalid data file !")
    return data

def verify_taxas(taxas : str, db_taxas : list):
    taxas = str.split(taxas, ',')
    for taxa in taxas:
        if taxa not in db_taxas:
            raise ValueError(f"One of the chosen classification taxa {taxas} is not present in the database!")
    return taxas


def verify_load_preclassified(data_file: Path):
    verify_file(data_file)
    preclassified = load_Xy_data(data_file)
    if not isinstance(preclassified['sequence'], list):
        raise ValueError("Invalid preclassified file !")
    elif not isinstance(preclassified['classification'], pd.DataFrame):
        raise ValueError("Invalid data file !")
    elif not isinstance(preclassified['classified_ids'], list):
        raise ValueError("Invalid data file !")
    elif not isinstance(preclassified['unknown_ids'], list):
        raise ValueError("Invalid data file !")
    return preclassified

# Saving
#########################################################################################################

def merge_classified_data(
    clf_data : dict,
    db_data: dict,
):
    # sequence
    sequence = db_data['sequence'].copy()
    sequence.extend(clf_data['sequence'])
    clf_data['sequence'] = sequence
    # classification
    classif = db_data['classification']
    clf_data['classification'] = classif.join(clf_data['classification'], how = 'outer', on = 'id')
    # classified_ids
    clf_ids = db_data['classified_ids'].copy()
    clf_ids.extend(clf_data['classified_ids'])
    clf_data['classified_ids'] = list(np.unique(clf_ids))
    # unknown_ids
    clf_ids = db_data['unknown_ids'].copy()
    clf_ids.extend(clf_data['unknown_ids'])
    clf_data['unknown_ids'] = list(np.unique(clf_ids))
    # classes
    dct_diff = {k : v for k,v in db_data.items() if k not in clf_data.keys()}
    clf_data = {**clf_data,**dct_diff}

    return clf_data

def merge_save_data(
    clf_data : dict,
    db_data : dict,
    end_taxa : str,
    outdir : Path,
    metagenome : str,
    preclassified : str = None,
):
    if preclassified is not None:
        clf_data = merge_classified_data(clf_data, preclassified)

    if end_taxa is not None:
        clf_data['sequence'] = clf_data['sequence'][:clf_data['sequence'].index(end_taxa)]

    clf_file = os.path.join(outdir, f'{metagenome}_classified.npz')
    save_Xy_data(clf_data, clf_file)
    
    return clf_data

def zip_X_y(X, y):
    num_blocks = X.num_blocks()
    len_x = X.count()
    if len_x > 1000:
        num_blocks = int(len_x / 50)
    ensure_length_ds(len_x, len(y))
    y = ray.data.from_arrow(pa.Table.from_pandas(y))
    X = X.repartition(len_x)
    y = y.repartition(len_x)
    for ds in [X, y]:
        if not ds.is_fully_executed():
            ds.fully_executed()
        df = X.zip(y).repartition(num_blocks)
    
    return df

def ensure_length_ds(len_x, len_y):
    if len_x != len_y:
        raise ValueError(
            'X and y have different lengths: {} and {}'.format(len_x, len_y))

# Datasets handling
#########################################################################################################

def convert_archaea_bacteria(df):
    df.loc[df['domain'].str.lower() == 'archaea', 'domain'] = 'Bacteria'
    return df

def verify_load_db(db_data):
    """
    Wrapper function for verifying and loading the db dataset
    """
    db_data = verify_load_data(db_data)
    files_lst = glob(os.path.join(db_data['profile'], '*.parquet'))
    db_ds = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))
    db_ds = db_ds.map_batches(convert_archaea_bacteria, batch_format = 'pandas')
    
    return db_data, db_ds

def verify_load_host_merge(db_data, host_data):
    """
    Wrapper function for verifying, loading and merging both datasets
    """
    db_data = verify_load_data(db_data)
    host_data = verify_load_data(host_data)
    verify_concordance_klength(len(db_data['kmers'][0]), len(host_data['kmers'][0]))
    merged_data, merged_ds = merge_db_host(db_data, host_data)
    
    return merged_data, merged_ds

def merge_db_host(db_data, host_data):
    """
    Merge the two databases along the rows axis
    """
    merged_db_host = {}
    merged_db_host_file = f"{db_data['profile']}_host_merged.npz"

    if os.path.exists(merged_db_host_file):
        merged_db_host = load_Xy_data(merged_db_host_file)
        files_lst = glob(os.path.join(merged_db_host['profile'], '*.parquet'))
        merged_ds = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))
    else:
        merged_db_host['profile'] = f"{db_data['profile']}_host_merged"
        files_lst = glob(os.path.join(db_data['profile'], '*.parquet'))
        db_ds = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))
        files_lst = glob(os.path.join(host_data['profile'], '*.parquet'))
        host_ds = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))

        cols2drop = [col for col in db_ds.schema().names if col not in ['id','domain',TENSOR_COLUMN_NAME]]
        db_ds = db_ds.drop_columns(cols2drop)

        cols2drop = [col for col in host_ds.schema().names if col not in ['id','domain',TENSOR_COLUMN_NAME]]
        host_ds = host_ds.drop_columns(cols2drop)

        merged_ds = db_ds.union(host_ds)
        merged_ds = merged_ds.map_batches(convert_archaea_bacteria, batch_format = 'pandas')
        merged_ds.write_parquet(merged_db_host['profile'])
    
    merged_db_host['ids'] = np.concatenate((db_data["ids"], host_data["ids"]))  # IDs
    merged_db_host['kmers'] = db_data['kmers']  # Features
    merged_db_host['taxas'] = ['domain']  # Known taxas for classification
    merged_db_host['fasta'] = (db_data['fasta'], host_data['fasta'])  # Fasta file needed for reads simulation
    
    save_Xy_data(merged_db_host, merged_db_host_file)

    return merged_db_host, merged_ds
