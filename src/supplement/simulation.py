#!/usr/bin python3

import modin.pandas as pd
import numpy as np

import os
import sys
import gzip
import tarfile
import argparse

from Bio import SeqIO, Entrez

__author__ = "Nicolas de Montigny"

__all__ = ['simulation','InSilicoSeq','get_args','fastq2fasta','write_cls_file']


def simulation(opt):
    abund_file = opt['prefix'] + '_abundance.txt'
    fastq_R1 = opt['prefix'] + '_R1.fastq.gz'
    fastq_R2 = opt['prefix'] + '_R2.fastq.gz'
    fasta_file = opt['prefix'] + '_data.fna.gz'
    cls_out_file = opt['prefix'] + '_class.csv'

    if not os.path.exists(os.path.dirname(opt['prefix'])):
        print('Output directory does not exists')
        sys.exit()

    if not os.path.isfile(fasta_file):
        InSilicoSeq(**opt)
        fastq2fasta(fastq_R1, fastq_R2, fasta_file)

    if not os.path.isfile(cls_out_file):
        write_cls_file(cls_out_file, opt['classes'], abund_file)

def InSilicoSeq(fasta, genomes, reads, type, prefix, **kwargs):
    # InSilicoSeq https://insilicoseq.readthedocs.io/en/latest/
    cmd = "iss generate -g {} -u {} -n {} --abundance halfnormal --model {} --output {} --compress --cpus {}".format(fasta,genomes,reads,type,prefix,len(os.sched_getaffinity(0)))
    os.system(cmd)

def fastq2fasta(fastq_R1, fastq_R2, fasta_file):
    with gzip.open(fastq_R1, "rt") as handle_R1, gzip.open(fastq_R2, "rt") as handle_R2, gzip.open(fasta_file, "at") as handle_out:
        for record_R1, record_R2 in zip(SeqIO.parse(handle_R1, 'fastq'), SeqIO.parse(handle_R2, 'fastq')):
            SeqIO.write(record_R1, handle_out, 'fasta')
            SeqIO.write(record_R2, handle_out, 'fasta')

def write_cls_file(cls_out_file, classes, abund_file):

    abund = list(pd.read_table(abund_file, header = None, names = ['id', 'abundance'])['id'])
    cls_in = pd.read_csv(classes)
    cls_out = pd.DataFrame(np.empty((len(abund),len(cls_in.columns))), columns = cls_in.columns)
    for i, id in enumerate(abund):
            row = cls_in.loc[cls_in['id'] == id].squeeze()
            cls_out.iloc[i] = row.copy()
    cls_out.to_csv(cls_out_file, index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate metagenomics sequencing reads using InSilicoSeq package')
    parser.add_argument('-f','--fasta', required=True, help='PATH to a fasta file containing bacterial genomes to build simulation from')
    parser.add_argument('-c','--classes', required=True, help='PATH to a csv file containing the classes associated to the fasta from which the simulation is built')
    parser.add_argument('-g','--genomes', type=int, default=100, help='Integer. The number of genomes to use for simulation')
    parser.add_argument('-a','--reads', type=int, default=50000, help='Integer. The number of reads to simulate')
    parser.add_argument('-t','--type', default='miseq', choices=['miseq','hiseq','novaseq'], help='Type of Illumina sequencing to be simulated among : MiSeq, HiSeq and NovaSeq')
    parser.add_argument('-p','--prefix', required=True, help='PATH to and filename prefix of outputed files')
    args = parser.parse_args()

    opt = vars(args)

    simulation(opt)
