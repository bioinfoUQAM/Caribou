#!/usr/bin python3
import os
import ray
import argparse

import numpy as np
import pandas as pd
import pyarrow as pa

from pathlib import Path
from shutil import rmtree

from utils import load_Xy_data, save_Xy_data
from joblib import Parallel, delayed, parallel_backend

def subset_expand_tensors(subset, lst_kmers):
    ray.data.set_progress_bars(False)
    lst_rows = []
    sub_cols = subset['kmers']
    sub_df = ray.data.read_parquet(subset['profile']) # distributed in parallel?
    for row in sub_df.iter_rows():
        row_full = np.zeros((1, len(lst_kmers)))
        row = row['__value__']
        for col in sub_cols:
            row_full[0, lst_kmers.index(col)] = row[sub_cols.index(col)]
        lst_rows.append(ray.put(row_full))
    ray.data.set_progress_bars(True)
    return lst_rows

# CLI
################################################################################
parser = argparse.ArgumentParser(
    description="This script merges subsets of a dataset from a list of data files into a single dataset")

parser.add_argument('-f', '--fasta', required=True, type=Path,
                    help='Path to the original fasta file')
parser.add_argument('-c', '--cls', required=True, type=Path,
                    help='Path to the original class file')
parser.add_argument('-l', '--list_file', required=True, type=Path,
                    help='Path to a .txt file containing one data file to merge per line')
parser.add_argument('-o', '--out', required=True, type=Path,
                    help='Name of a .npz file to output the merged dataset')
args = parser.parse_args()

opt = vars(args)

# Verify folder
################################################################################
files_lst = []
if not os.path.isfile(opt['list_file']):
    raise ValueError('Cannot find file {}'.format(opt['list_file']))
else:
    with open(opt['list_file'], 'r') as f:
        for line in f:
            line = line.strip()
            if os.path.isfile(line):
                files_lst.append(line)
            else:
                raise ValueError('Cannot find data file {}'.format(line))
if not os.path.isfile(opt['fasta']):
    raise ValueError('Cannot find file {}'.format(opt['fasta']))
if not os.path.isfile(opt['cls']):
    raise ValueError('Cannot find file {}'.format(opt['cls']))
if os.path.splitext(opt['cls'])[1] != '.csv':
    raise ValueError('Class file must be a .csv file')
if os.path.splitext(opt['out'])[1] != '.npz':
    raise ValueError('Output file must be a .npz file')

# Inits
################################################################################
ray.init()

data = {}

subsets = []

lst_profiles = []
lst_ids = []
lst_classes = []
lst_kmers = []
lst_taxas = []

tmp_dir = os.path.join(os.path.dirname(opt['out']), 'tmp')

rows_full = []
merged_profile_df = None
merged_profile_file = os.path.splitext(opt['out'])[0]

merged_cls_df = None

ids = None
cls = None

# Merge datasets
################################################################################
# Load data from each files
with parallel_backend('threading'):
    subsets = Parallel(n_jobs=-1, verbose=1)(delayed(load_Xy_data)(file) for file in files_lst)

# Extract data per file in lists
for subset in subsets:
    lst_profiles.append(subset['profile'])
    lst_ids.extend(subset['ids'])
    lst_classes.extend(subset['classes'])
    lst_kmers.append(subset['kmers'])
    lst_taxas.append(subset['taxas'])

# Flatten lists that must be flattened
lst_kmers = list(np.unique(np.concatenate(lst_kmers)))
lst_taxas = list(np.unique(np.concatenate(lst_taxas)))

# Merge profiles
# for subset in subsets:
#     rows_full.extend(subset_expand_tensors(subset, lst_kmers))
with parallel_backend('threading'):
    rows_full = Parallel(n_jobs=-1, verbose=1)(delayed(subset_expand_tensors)(subset, lst_kmers) for subset in subsets)

# Build merged profile dataframe
merged_profile_df = ray.data.from_numpy_refs(list(np.concatenate(rows_full)))
# Add ID column
num_blocks = merged_profile_df.num_blocks()
merged_profile_df = merged_profile_df.repartition(
    merged_profile_df.count()).zip(
        ray.data.from_arrow(
            pa.Table.from_pandas(
                pd.DataFrame({
                    'id': lst_ids,
                })
            )
        ).repartition(merged_profile_df.count())
    ).repartition(num_blocks)
# Write new profile to file
merged_profile_df.write_parquet(merged_profile_file)

# Generate classes array
ids = pd.DataFrame({'id': lst_ids})
cls = pd.read_csv(opt['cls'])
cls = pd.merge(cls, ids, on='id', how='inner')
cls = cls.drop(columns=['id'])

# Save merged dataset
################################################################################
data['profile'] = merged_profile_file  # Kmers profile
data['classes'] = np.array(cls)  # Class labels
data['kmers'] = lst_kmers  # Features
data['taxas'] = lst_taxas  # Known taxas for classification
data['fasta'] = opt['fasta']  # Fasta file -> simulate reads if cv

save_Xy_data(data, opt['out'])

# Delete subset files
################################################################################
# for file in files_lst:
#     os.remove(file)
# for subset in subsets:
#     rmtree(subset['profile'])

# Recreate k-mers list file
################################################################################
with open(os.path.join(os.path.dirname(opt['out']), 'kmers_list.txt'), 'w') as handle:
    handle.writelines("%s\n" % item for item in lst_kmers)

print('Datasets merged successfully')
