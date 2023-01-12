#!/usr/bin python3

import ray
import json
import argparse

from utils import *
from pathlib import Path
from os.path import dirname
from outputs.out import Outputs

__author__ = 'Nicolas de Montigny'

__all__ = ['out_2_user']

# Initialisation / validation of parameters from CLI
################################################################################
def out_2_user(opt):
    data_bacteria = verify_load_data(opt['data_bacteria'])
    classified_data = verify_load_classified(opt['classified_data'])
    out_dir = dirname(opt['classified_data'])

    outs = Outputs(data_bacteria,
        out_dir,
        len(data_bacteria['kmers'][0]),
        opt['model_type'],
        opt['dataset_name'],
        opt['host_name'],
        classified_data
    )

    if opt['mpa']:
        outs.abundances()
    if opt['kronagram']:
        outs.kronagram()
    if opt['report']:
        outs.report()
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
    parser.add_argument('-cd','--classified_data', required=True, type=Path, help='PATH to a npz file containing the data classified by Caribou')
    parser.add_argument('-model','--model_type', required=True, choices=['sgd','mnb','lstm_attention','cnn','widecnn'], help='The type of model used for classification')
    parser.add_argument('-dt','--dataset_name', required=True, help='Name of the classified dataset used to name files')
    parser.add_argument('-dh','--host_name', default=None, help='Name of the host database used to name files')
    parser.add_argument('-m','--mpa', action='store_true', help='Should the mpa-style output be generated?')
    parser.add_argument('-k','--kronagram', action='store_true', help='Should the interactive kronagram be generated?')
    parser.add_argument('-r','--report', action='store_true', help='Should the abundance report be generated?')
    parser.add_argument('-wd','--workdir', default='/tmp/spill', type=Path, help='Optional. Path to a working directory where tuning data will be spilled')
    # parser.add_argument('-b', '--biom', action='store_true', help='Should the biom file be generated?')
    args = parser.parse_args()

    opt = vars(args)

    ray.init(
        _system_config = {
            'object_spilling_config': json.dumps(
                {'type': 'filesystem', 'params': {'directory_path': str(opt['workdir'])}})
        }
    )

    out_2_user(opt)