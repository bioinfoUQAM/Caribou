#!/usr/bin python3

import os
import bz2
import gzip
import zipfile
import argparse

import pandas as pd

from Bio import SeqIO
from pathlib import Path

def verify_csv(file):
    ext = os.path.splitext(file)[1]
    if ext != '.csv':
        raise ValueError('Classes file must be in CSV format!')

def verify_extract_ids(file):
    ids = []
    ext = os.path.splitext(file)[1]
    if ext in ['.fa','.fna','.fasta']:
        with open(file, 'rt') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                ids.append(record.id)
    elif ext in ['.zip']:
        with zipfile.open(file, 'rt') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                ids.append(record.id)
    elif ext in ['.gz','.gzip']:
        with gzip.open(file, 'rt') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                ids.append(record.id)
    elif ext in ['.bz','.bzip']:
        with bz2.open(file, 'rt') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                ids.append(record.id)
    else:
        raise ValueError('Unknown file extension! Extension should be fasta, zip, gzip or bzip2. Yout file extension "{}" is not known.'.format(ext))

    return ids

# CLI
################################################################################
parser = argparse.ArgumentParser(description="This script subsets a classes.csv file according to the ids found in the supplied fasta file")

parser.add_argument('-f','--fasta', required=True, type=Path, help='Path to a fasta file from which the sequence ids will be extracted. File extension can be .fa, .fasta, .fna and their zipped, gzipped and bzipped2 equivalents')
parser.add_argument('-c','--classes', required=True, type=Path, help='Path to a csv file from which the subset of classes will be extracted')
parser.add_argument('-o','--output', required=True, type=Path, help='Path to the subset csv file which will be outputed')
args = parser.parse_args()

opt = vars(args)

# Verification of files
################################################################################
if not os.path.isfile(opt['fasta']):
    raise ValueError('Cannot find file {}'.format(opt['fasta']))
if not os.path.isfile(opt['classes']):
    raise ValueError('Cannot find file {}'.format(opt['classes']))

verify_csv(opt['classes'])
verify_csv(opt['output'])

# Parse ids list
################################################################################
ids = verify_extract_ids(opt['fasta'])
ids = pd.DataFrame({'id' : ids})

# Extract classes from csv into a new pandas
################################################################################
cls = pd.read_csv(opt['classes'])

# Merge and save the classes subset file
################################################################################
cls_out = pd.merge(cls,ids, on = 'id', how = 'inner')
cls_out = cls_out.drop_duplicates(subset = 'id', keep = 'first')

cls_out.to_csv(opt['output'], index = False)
