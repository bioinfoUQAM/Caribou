#!/usr/bin/python3

from Caribou.data.build_data import build_load_save_data

import pandas as pd

from tensorflow.compat.v1 import ConfigProto, Session
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.config import list_physical_devices

import sys
import os.path
import argparse

from os import makedirs

__author__ = "Nicolas de Montigny"

__all__ = ['caribou']

# GPU & CPU setup
################################################################################
gpus = list_physical_devices('GPU')
if gpus:
    config = ConfigProto(device_count={'GPU': len(gpus), 'CPU': os.cpu_count()})
    sess = Session(config=config)
    set_session(sess);

# Part 0 - Initialisation / extraction of parameters from command line
################################################################################
def caribou(opt):

    # Folders creation for output
    outdirs = {}
    outdirs["main_outdir"] = opt['outdir']
    outdirs["data_dir"] = os.path.join(outdirs["main_outdir"], "data")
    makedirs(outdirs["main_outdir"], mode=0o700, exist_ok=True)
    makedirs(outdirs["data_dir"], mode=0o700, exist_ok=True)

# Part 1 - K-mers profile extraction
################################################################################

    if opt['cls_file'] is not None and kmers_list is None:
        # Reference Database Only
        k_profile_database = build_load_save_data((opt['seq_file'], opt['cls_file']),
            "none",
            outdirs["data_dir"],
            opt['dataset_name'],
            None,
            k = opt['k_length'],
        )

        # Save kmers list to file for further extractions
        with open(os.path.join(outdirs["data_dir"],'kmers_list.txt'),'w') as handle:
            handle.writelines("%s\n" % item for item in k_profile_database['kmers_list'])

        print("Caribou finished extracting k-mers of {}".format(opt['dataset_name']))

    elif opt['cls_file'] is None and opt['kmers_list'] is not None:
        # Read kmers file to put in list
        kmers_list = []
        with open(opt['kmers_list'], 'r') as handle:
            kmers_list = [kmer.rstrip() for kmer in handle.readlines()]

        # Metagenome to analyse
        k_profile_metagenome = build_load_save_data(opt['seq_file'],
            "none",
            outdirs["data_dir"],
            opt['dataset_name'],
            None,
            k = opt['k_length'],
            kmers_list = kmers_list
        )
        print("Caribou finished extracting k-mers of {}".format(opt['dataset_name']))

    else:
        print("Caribou cannot extract k-mers because there is no class file or k-mers list given")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract k-mers of one dataset and save it to drive')
    parser.add_argument('-seq','--seq_file', required=True, help='PATH to a fasta file containing bacterial genomes to build k-mers from')
    parser.add_argument('-cls','--cls_file', default=None, help='PATH to a csv file containing classes of the corresponding fasta')
    parser.add_argument('-dt','--dataset_name', required=True, help='Name of the dataset used to name files')
    parser.add_argument('-k','--k_length', required=True, help='Length of k-mers to extract')
    parser.add_argument('-l','--kmers_list', default=None, help='PATH to a file containing a list of k-mers to be extracted if the dataset is not a training database')
    parser.add_argument('-o','--outdir', required=True, help='PATH to a directory on file where outputs will be saved')
    args = parser.parse_args()

    opt = vars(args)

    caribou(opt)
