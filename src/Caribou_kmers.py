#!/usr/bin python3

import ray
import json
import os.path
import argparse
import numpy as np

from utils import *
from time import time
from pathlib import Path
from data.build_data import build_load_save_data

__author__ = "Nicolas de Montigny"

__all__ = ['kmers_dataset']

"""
This script extracts K-mers of the given dataset using the available ressources on the computer before saving it to drive.
"""

# Initialisation / validation of parameters from CLI
################################################################################
def kmers_dataset(opt):
    kmers_list = None

    # Verify if there are files to analyse
    verify_seqfiles(opt['seq_file'], opt['seq_file_host'])

    # Verification of existence of files
    for file in [opt['cls_file'],opt['cls_file_host'],opt['kmers_list']]:
        verify_file(file)

    # Verification of k length
    opt['k_length'], kmers_list = verify_kmers_list_length(opt['k_length'], opt['kmers_list'])

    # Verify path for saving
    outdirs = define_create_outdirs(opt['outdir'])
    
    # Initialize cluster
    ray.init(
        _system_config = {
            'object_spilling_config': json.dumps(
                {'type': 'filesystem', 'params': {'directory_path': str(opt['workdir'])}})
        }
    )

# K-mers profile extraction
################################################################################

    if kmers_list is None:
        t_start = time()
        # Reference Database Only
        if opt['seq_file'] is not None and opt['cls_file'] is not None and opt['seq_file_host'] is None and opt['cls_file_host'] is None:
            k_profile_database = build_load_save_data((opt['seq_file'],opt['cls_file']),
                None,
                outdirs["data_dir"],
                opt['dataset_name'],
                opt['host_name'],
                k = opt['k_length'],
                kmers_list = None,
            )

            # Save kmers list to file for further extractions
            kmers_list = k_profile_database['kmers']
            with open(os.path.join(outdirs["data_dir"],'kmers_list.txt'),'w') as handle:
                handle.writelines("%s\n" % item for item in kmers_list)
            t_end = time()
            t_kmers = t_end - t_start

            print(f"Caribou finished extracting k-mers of {opt['dataset_name']} in {t_kmers} seconds.")

        # Reference database and host
        elif opt['seq_file'] is not None and opt['cls_file'] is not None and opt['seq_file_host'] is not None and opt['cls_file_host'] is not None:

            t_start = time()
            k_profile_database, k_profile_host  = build_load_save_data((opt['seq_file'],opt['cls_file']),
                (opt['seq_file_host'],opt['cls_file_host']),
                outdirs["data_dir"],
                opt['dataset_name'],
                opt['host_name'],
                k = opt['k_length'],
                kmers_list = None,
            )

            # Save kmers list to file for further extractions
            kmers_list = k_profile_database['kmers']
            with open(os.path.join(outdirs["data_dir"],'kmers_list.txt'),'w') as handle:
                handle.writelines("%s\n" % item for item in kmers_list)
            t_end = time()
            t_kmers = t_end - t_start

            print(f"Caribou finished extracting k-mers of {opt['dataset_name']} and {opt['host_name']} in {t_kmers} seconds.")
    else:
        # Reference Host only
        if opt['seq_file'] is not None and opt['cls_file'] is not None:

            t_start = time()
            k_profile_host = build_load_save_data(None,
            (opt['seq_file'],opt['cls_file']),
            outdirs["data_dir"],
            None,
            opt['host_name'],
            k = opt['k_length'],
            kmers_list = kmers_list
            )
            t_end = time()
            t_kmers = t_end - t_start
            print(f"Caribou finished extracting k-mers of {opt['host_name']} in {t_kmers} seconds.")

        # Dataset to analyse only
        elif opt['seq_file'] is not None and opt['cls_file'] is None:

            t_start = time()
            k_profile_metagenome = build_load_save_data(opt['seq_file'],
            None,
            outdirs["data_dir"],
            opt['dataset_name'],
            None,
            k = opt['k_length'],
            kmers_list = kmers_list
            )
            t_end = time()
            t_kmers = t_end - t_start
            
            print(f"Caribou finished extracting k-mers of {opt['dataset_name']} in {t_kmers} seconds.")

        else:
            raise ValueError(
                "Caribou cannot extract k-mers because there are missing parameters !\n" +
                "Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki")

# Argument parsing from CLI
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script extracts K-mers of the given dataset using the available ressources on the computer before saving it to drive.')
    # Database
    parser.add_argument('-s','--seq_file', default=None, type=Path, help='PATH to a fasta file containing bacterial genomes to build k-mers from \
        or a folder containing fasta files with one sequence per file')
    parser.add_argument('-c','--cls_file', default=None, type=Path, help='PATH to a csv file containing classes of the corresponding fasta')
    parser.add_argument('-dt','--dataset_name', default='dataset', help='Name of the dataset used to name files')
    # Host
    parser.add_argument('-sh','--seq_file_host', default=None, type=Path, help='PATH to a fasta file containing host genomes to build k-mers from \
        or a folder containing fasta files with one sequence per file')
    parser.add_argument('-ch','--cls_file_host', default=None, type=Path, help='PATH to a csv file containing classes of the corresponding host fasta')
    parser.add_argument('-dh','--host_name', default='host', help='Name of the host used to name files')
    # Parameters
    parser.add_argument('-k','--k_length', required=True, type=int, help='Length of k-mers to extract')
    parser.add_argument('-l','--kmers_list', default=None, type=Path, help='PATH to a file containing a list of k-mers to be extracted if the dataset is not a training database')
    parser.add_argument('-o','--outdir', required=True, type=Path, help='PATH to a directory on file where outputs will be saved')
    parser.add_argument('-wd','--workdir', default='/tmp/spill', type=Path, help='Optional. Path to a working directory where tuning data will be spilled')
    args = parser.parse_args()

    opt = vars(args)

    kmers_dataset(opt)
