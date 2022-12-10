#!/usr/bin python3

import argparse

from utils import *
from pathlib import Path
from outputs.out import Outputs
from os.path import dirname

__author__ = 'Nicolas de Montigny'

__all__ = ['caribou']

# Initialisation / validation of parameters from CLI
################################################################################
def out_2_user():
    data_bacteria = verify_load_data(opt['data_bacteria'])
    classified_data = verify_load_data(opt['classified_data'])
    out_dir = dirname(opt['classified_data'])

    outs = Outputs(data_bacteria,
        out_dir,
        len(data_bacteria['kmers'][0]),
        opt['model_type'],
        opt['dataset_name'],
        opt['host_name'],
        classified_data
    )

    if opt['abundance']:
        outs.abundance_table()
    if opt['kronagram']:
        outs.kronagram()
    if opt['report']:
        outs.report()
    if opt['fasta']:
        outs.fasta()
    # if opt['biom']:
    #     outs.biom()

    print('Outputs generated with success')
# 
################################################################################

# Argument parsing from CLI
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script produces outputs from the results of classified data by Caribou.')
    parser.add_argument('-db','--data_bacteria', required=True, type=Path, help='PATH to a npz file containing the data corresponding to the k-mers profile for the bacteria database')
    parser.add_argument('-clf','--classified_data', required=True, type=Path, help='PATH to a npz file containing the data classified by Caribou')
    parser.add_argument('-model','--model_type', required=True, choices=['sgd','mnb','lstm_attention','cnn','widecnn'], help='The type of model used for classification')
    parser.add_argument('-dt','--dataset_name', required=True, help='Name of the classified dataset used to name files')
    parser.add_argument('-ds','--host_name', default=None, help='Name of the host database used to name files')
    parser.add_argument('-a','--abundance', action='store_true', help='Should the abundance table be generated?')
    parser.add_argument('-k','--kronagram', action='store_true', help='Should the interactive kronagram be generated?')
    parser.add_argument('-r','--report', action='store_true', help='Should the full report be generated?')
    parser.add_argument('-f', '--fasta', action='store_true', help='Should the fasta file per classified taxa be generated?')
    # parser.add_argument('-b', '--biom', action='store_true', help='Should the biom file be generated?')
    args = parser.parse_args()

    opt = vars(args)

    out_2_user(opt)