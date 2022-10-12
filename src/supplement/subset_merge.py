#!/usr/bin python3

import os
import ray
import argparse

import numpy as np

from glob import glob
from pathlib import Path
from shutil import rmtree

from utils import load_Xy_data, save_Xy_data

# Functions
################################################################################
def batch_read_write(batch, dir):
    df = ray.data.read_parquet(batch)
    df.write_parquet(dir)
    for file in batch:
        os.remove(file)

# CLI
################################################################################
parser = argparse.ArgumentParser(
    description="This script merges subsets of a dataset from one folder into a single dataset")

parser.add_argument('-f', '--fasta', required=True, type=Path,
                    help='Path to the original fasta file')
parser.add_argument('-d', '--dir', required=True, type=Path,
                    help='Path to a folder containing all the subsets to merge')
parser.add_argument('-o', '--out', required=True, type=Path,
                    help='Name of a .npz file to output the merged dataset')
args = parser.parse_args()

opt = vars(args)

# Verify folder
################################################################################
if not os.path.isdir(opt['dir']):
    raise ValueError('Cannot find folder {}'.format(opt['dir']))
if not os.path.isfile(opt['fasta']):
    raise ValueError('Cannot find file {}'.format(opt['fasta']))
if os.path.splitext(opt['out'])[1] != '.npz':
    raise ValueError('Output file must be a .npz file')

# Inits
################################################################################
ray.init()

data = {}

subsets = []
list_sub_dir = []

list_profiles = []
classes = []
list_kmers = []
list_taxas = []
list_fasta = []

empty = True

opt = {'fasta' : '/mnt/GTDB.fna.gz', 'dir' : '/mnt/output/data/','out' : '/mnt/output/data/Xy_genome_GTDB_data_K20.npz'}

parent_dir = os.path.split(opt['out'])[0]
ds_dir = os.path.splitext(opt['out'])[0]

# Merge datasets
################################################################################
# List subsets
subsets = glob(os.path.join(opt['dir'], '*.npz'))

# Loop over subsets to append to lists
for subset in subsets:
    # Load data
    data = load_Xy_data(subset)
    # Append to lists
    if empty is True:
        list_sub_dir.append(data['profile'])
        list_profiles = np.array(glob(os.path.join(data['profile'],'*.parquet')))
        list_taxas = np.array(data['taxas'])
        empty = False
    else:
        list_profiles = np.append(list_profiles, np.array(glob(os.path.join(data['profile'],'*.parquet'))))

list_profiles = list(list_profiles)
# Read/concatenate files with Ray by batches
nb_batch = 0
while np.ceil(len(list_profiles)/1000) > 1:
    batches_list = np.array_split(list_profiles, np.ceil(len(list_profiles)/1000))
    batch_dir = os.path.join(parent_dir, 'batch_{}'.format(nb_batch))
    os.mkdir(batch_dir)
    for batch in batches_list:
        batch_read_write(list(batch), batch_dir)
    list_profiles = glob(os.path.join(batch_dir, '*.parquet'))
    nb_batch += 1

# Read/concatenate batches and save with Ray
df = ray.data.read_parquet(list_profiles)
df.write_parquet(ds_dir)

# Extract kmers list

# Generate classes array

# Save merged dataset
################################################################################
data['profile'] = ds_dir  # Kmers profile
data['classes'] = list_classes  # Class labels
data['kmers'] = list_kmers  # Features
data['taxas'] = list_taxas  # Known taxas for classification
data['fasta'] = opt['fasta']  # Fasta file -> simulate reads if cv

save_Xy_data(data, opt['out'])

# Delete subset files
################################################################################
for file in subsets:
    os.remove(file)
for dir in list_sub_dir:
    rmtree(dir)

# Recreate k-mers list file
################################################################################
kmers_list = data['kmers']
with open(os.path.join(parent_dir, 'kmers_list.txt'), 'w') as handle:
    handle.writelines("%s\n" % item for item in kmers_list)

print('Datasets merged successfully')