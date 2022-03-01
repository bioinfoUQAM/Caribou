#!/usr/bin/python3

from Caribou.data.build_data import build_load_save_data

import pandas as pd

from tensorflow.compat.v1 import ConfigProto, Session
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.config import list_physical_devices

import sys
import os.path
import argparse
import pathlib

from os import makedirs

__author__ = "Nicolas de Montigny"

__all__ = ['kmers_dataset']

"""
This script extracts K-mers of the given dataset using the available ressources on the computer before saving it to drive.
"""

# GPU & CPU setup
################################################################################
gpus = list_physical_devices('GPU')
if gpus:
    config = ConfigProto(device_count={'GPU': len(gpus), 'CPU': os.cpu_count()})
    sess = Session(config=config)
    set_session(sess);

# Initialisation / validation of parameters from CLI
################################################################################
def kmers_dataset(opt):

    # Verification of existence of files
    for file in [opt['seq_file'],opt['cls_file']]:
        if not os.path.isfile(file):
            print("Cannot find file {} ! Exiting".format(file))
            sys.exit()

    # Verification of k length
    if opt['k_length'] <= 0:
        print("Invalid K-mers length ! Exiting")
        sys.exit()

    # Verify path for saving
    outdir_path, outdir_folder = os.path.split(opt['outdir'])
    if not os.path.isdir(outdir) and os.path.exists(outdir_path):
        print("Created output folder")
        os.makedirs(outdir)
    elif not os.path.exists(outdir_path):
        print("Cannot find where to create output folder ! Exiting")
        sys.exit()

    # Folders creation for output
    outdirs = {}
    outdirs["main_outdir"] = opt['outdir']
    outdirs["data_dir"] = os.path.join(outdirs["main_outdir"], "data")
    makedirs(outdirs["main_outdir"], mode=0o700, exist_ok=True)
    makedirs(outdirs["data_dir"], mode=0o700, exist_ok=True)

# K-mers profile extraction
################################################################################

    if opt['cls_file'] is not None and opt['kmers_list'] is None:
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


# Argument parsing from CLI
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script extracts K-mers of the given dataset using the available ressources on the computer before saving it to drive.')
    parser.add_argument('-seq','--seq_file', required=True, type=pathlib.Path, help='PATH to a fasta file containing bacterial genomes to build k-mers from')
    parser.add_argument('-cls','--cls_file', required=True, type=pathlib.Path, help='PATH to a csv file containing classes of the corresponding fasta')
    parser.add_argument('-dt','--dataset_name', required=True, help='Name of the dataset used to name files')
    parser.add_argument('-k','--k_length', required=True, type=int, help='Length of k-mers to extract')
    parser.add_argument('-l','--kmers_list', default=None, type=pathlib.Path, help='PATH to a file containing a list of k-mers to be extracted if the dataset is not a training database')
    parser.add_argument('-o','--outdir', required=True, type=pathlib.Path, help='PATH to a directory on file where outputs will be saved')
    args = parser.parse_args()

    opt = vars(args)

    kmers_dataset(opt)
