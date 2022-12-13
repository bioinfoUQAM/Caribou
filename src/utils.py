import os
import numpy as np
import pandas as pd

from pathlib import Path
from warnings import warn

__author__ = "Nicolas de Montigny"

__all__ = [
    'load_Xy_data',
    'save_Xy_data',
    'verify_file',
    'verify_data_path',
    'verify_saving_path',
    'verify_host',
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
    'verify_load_classified'
]

# Data handling
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
        raise ValueError('Cannot find file {} !'.format(file))

def verify_data_path(dir : Path):
    if not os.path.exists(dir):
        raise ValueError("Cannot find data folder {} ! Exiting".format(dir))

def verify_saving_path(dir : Path):
    path, folder = os.path.split(dir)
    if not os.path.exists(path):
        raise ValueError("Cannot find where to create output folder !")

def verify_host(host : str):
    if host not in ['none', 'None', None]:
        return host
    else:
        return None

def verify_boolean(val : bool, parameter : str):
    if val not in [True,False,None]:
        raise ValueError(
            'Invalid value for {} ! Please use boolean values !\n'.format(parameter) +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')

def verify_positive_int(val : int, parameter : str):
    if type(val) != int or val < 0:
        raise ValueError(
            'Invalid value for {} ! Please use a positive integer !\n'.format(parameter) +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')

def verify_0_1(val : float, parameter : str):
    if type(val) != float:
        raise ValueError(
            'Invalid value for {} ! Please use a float between 0 and 1 !\n'.format(parameter) +
            'Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki')
    elif not 0 <= val <= 1:
        raise ValueError(
            'Invalid value for {} ! Please use a float between 0 and 1 !\n'.format(parameter) +
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
        raise ValueError("No file to extract K-mers from !")

def verify_concordance_klength(klen1 : int, klen2 : int):
    if klen1 != klen2:
        raise ValueError("K length between datasets is inconsistent ! Exiting\n" +
                "K length of bacteria dataset is {} while K length from host is {}").format(klen1, klen2)

# Verif + handling
#########################################################################################################

def define_create_outdirs(dir : Path):
    outdirs = {}
    verify_saving_path(dir)
    outdirs['main_outdir'] = dir
    outdirs['data_dir'] = os.path.join(dir, 'data')
    outdirs['models_dir'] = os.path.join(dir, 'models')
    outdirs['results_dir'] = os.path.join(dir, 'results')
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

def verify_load_classified(classified_data: Path):
    verify_file(classified_data)
    data = load_Xy_data(classified_data)
    if len(data['sequence']) == 0:
        raise ValueError("No classified taxa present in data file !")
    elif len(data['sequence']) != len(data.keys())-1:
        raise ValueError("Inconsistent number of classified data vs metadata in data file !")
    else:
        for taxa in data['sequence']:
            verify_data_path(data[taxa]['profile'])
        if not isinstance(data[taxa]['ids'], list):
            raise ValueError("Invalid classified data file !")
        elif not isinstance(data[taxa]['kmers'], list):
            raise ValueError("Invalid classified data file !")
        elif not isinstance(data[taxa]['classification'], pd.DataFrame):
            raise ValueError("Invalid classified data file !")
        elif not isinstance(data[taxa]['classified_ids'], list):
            raise ValueError("Invalid classified data file !")
            
    return data


def verify_taxas(taxas : str, db_taxas : list):
    taxas = str.split(taxas, ',')
    for taxa in taxas:
        if taxa not in db_taxas:
            raise ValueError("One of the chosen classification taxa {} is not present in the database!".format(taxas))
    return taxas
    