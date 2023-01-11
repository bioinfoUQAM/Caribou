#!/usr/bin python3

import pandas as pd

import os
import ray
import gzip
import argparse

from utils import *
from Bio import SeqIO
from glob import glob
from pathlib import Path
from models.reads_simulation import readsSimulation

__author__ = "Nicolas de Montigny"

__all__ = ['simulation','extract_genomes']


def simulation(opt):
    # Validation of parameters
    verify_file(opt['fasta'])
    verify_file(opt['classes'])
    if opt['kmers_length'] is not None and opt['kmers_list'] is not None:
        opt['kmers_length'], opt['kmers_list'] = verify_kmers_list_length(opt['kmers_length'], opt['kmers_list'])
    
    print(opt['kmers_length'])
    print(len(opt['kmers_list']))
    
    # Prepare for simulation
    genomes = extract_genomes(opt['fasta'])
    cls_df = pd.read_csv(opt['classes'])

    ray.init()
    # Execute simulation with k-mers extraction
    if opt['kmers_length'] is not None and opt['kmers_list'] is not None:
        outdirs = define_create_outdirs(opt['outputs'])
        data = readsSimulation(
            opt['fasta'],
            cls_df,
            genomes,
            opt['type'],
            outdirs['data_dir'],
            opt['dataset_name']
        ).simulation(opt['kmers_length'], opt['kmers_list'])
        files2remove = glob(os.path.join(outdirs['data_dir'], '*tmp*'))
        files2remove.extend(glob(os.path.join(outdirs['data_dir'], '*fastq*')))
        print('Reads Simulation and k-mers extraction done. Data saved in {}'.format(outdirs['data_dir']))

    else:
    # Execute simulation alone
        verify_saving_path(opt['outputs'])
        readsSimulation(
            opt['fasta'],
            cls_df,
            genomes,
            opt['type'],
            opt['outputs'],
            opt['dataset_name']
        ).simulation()
        files2remove = glob(os.path.join(opt['outputs'], '*tmp*'))
        files2remove.extend(glob(os.path.join(opt['outputs'], '*fastq*')))
        print('Reads Simulation done. Data saved in {}'.format(opt['outputs']))
    
    for file in files2remove:
        os.remove(file)

def extract_genomes(fasta):
    genomes = []
    if os.path.splitext(fasta)[1] == '.gz':
        with gzip.open(fasta, 'rt') as f:
            for record in SeqIO.parse(f, "fasta"):
                genomes.append(record.id)
    else:
        with open(fasta) as f:
            for record in SeqIO.parse(f, "fasta"):
                genomes.append(record.id)
    return genomes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate metagenomics sequencing reads using InSilicoSeq package')
    parser.add_argument('-f','--fasta', required=True, help='PATH to a fasta file containing bacterial genomes to build simulation from')
    parser.add_argument('-c','--classes', required=True, help='PATH to a csv file containing the classes associated to the fasta from which the simulation is built')
    parser.add_argument('-dt','--dataset_name', required=True, help='Name of the dataset used to name files')
    parser.add_argument('-k','--kmers_length', type=int, default=None, help='Optional. Length of k-mers to be extracted after the simulation')
    parser.add_argument('-l','--kmers_list', type=Path, default=None, help='Optional. PATH to a file containing a list of k-mers to be extracted after the simulation. Should be the same as the reference database')
    parser.add_argument('-t','--type', default='miseq', choices=['miseq','hiseq','novaseq'], help='Type of Illumina sequencing to be simulated among : MiSeq, HiSeq and NovaSeq')
    parser.add_argument('-o','--outputs', required=True, help='PATH to and filename prefix of outputed files')
    args = parser.parse_args()

    opt = vars(args)

    simulation(opt)
