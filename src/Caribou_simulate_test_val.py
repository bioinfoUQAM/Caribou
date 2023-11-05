#!/usr/bin python3

import argparse

from utils import *
from time import time
from pathlib import Path

__author__ = "Nicolas de Montigny"

__all__ = ['simulation']

"""
This script simulate sequencing reads for validation and/or testing dataset(s) from a whole genome dataset
The dataset should be in the form of a k-mers counts matrix and could have the k-mers reduced as well
The script leverages the InSilicoSeq package for simulation of sequencing reads
"""

# Initialisation / validation of parameters from CLI
################################################################################
def simulation(opt):
    """
    1. Verify existence of files and load data
    2. Verify k-mers length concordance
    3. Initialize cluster
    """
    if opt['hostset'] is not None:
        db_data, db_ds = verify_load_host_merge(opt['dataset'], opt['hostset'])
    else:
        db_data, db_ds = verify_load_db(opt['dataset'])
        
    verify_file(opt['kmers_list'])
    
    outdirs = define_create_outdirs(opt['outdir'])
    
    init_ray_cluster(opt['workdir'])

# Dataset(s) simulation
################################################################################
    """
    1. Verify the datasets to simulate
    2. Split the database dataset (possibly merged) into required dataset
    3. Run the simulation for each dataset required
    """
    t_test = None
    t_val = None
    if opt['test']:
        t_s = time()
        test_ds = split_dataset(db_ds, db_data, 'test')
        if test_ds is not None:
            sim_dataset(test_ds, db_data, 'test')
        t_test = time() - t_s
    if opt['validation']:
        t_s = time()
        val_ds = split_dataset(db_ds, db_data, 'validation')
        if val_ds is not None:
            sim_dataset(val_ds, db_data, 'validation')
        t_val = time() - t_s
    
    if t_test is not None:
        print(f'Caribou finished generating the test dataset in {t_test} seconds')
    if t_val is not None:
        print(f'Caribou finished generating the validation dataset simulated in {t_val} seconds')
    
# Argument parsing from CLI
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script simulate sequencing reads for validation and/or testing dataset(s) from a whole genome dataset')
    # Database
    parser.add_argument('-db','--dataset', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the bacteria database')
    parser.add_argument('-dt','--dataset_name', default='dataset', help='Name of the dataset used to name files')
    # Host
    parser.add_argument('-dh','--hostset', default=None, type=Path, help='Path to .npz data for extracted k-mers profile of host')
    parser.add_argument('-ds','--hostset_name', default=None, help='Name of the host database used to name files')
    # Simulation flags
    parser.add_argument('-v', '--validation', action='store_true', help='Flag argument for making a "validation"-named simulated dataset')
    parser.add_argument('-t', '--test', action='store_true', help='Flag argument for making a "test"-named simulated dataset')
    # Parameters
    parser.add_argument('-l','--kmers_list', type=Path, default=None, help='Optional. PATH to a file containing a list of k-mers to be extracted after the simulation. Should be the same as the reference database')
    parser.add_argument('-o','--outdir', required=True, type=Path, help='Path to folder for outputing tuning results')
    parser.add_argument('-wd','--workdir', default='/tmp/spill', type=Path, help='Optional. Path to a working directory where tuning data will be spilled')
    args = parser.parse_args()

    opt = vars(args)

    if not opt['test'] and not opt['validation']:
        raise ValueError('Missing flags for datasets to simulate, please use the -v and/or -t flags to decide which dataset to generate.')
    else:
        simulation(opt)