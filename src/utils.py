import numpy as np
import pandas as pd

import os
import ray

__author__ = "Nicolas de Montigny"

__all__ = ['load_Xy_data','save_Xy_data','merge_database_host']

# Load data from file
def load_Xy_data(Xy_file):
    with np.load(Xy_file, allow_pickle=True) as f:
        return f['data'].tolist()

# Save data to file
def save_Xy_data(df, Xy_file):
    np.savez(Xy_file, data = df)

# Merge database and host data
def merge_database_host(database_data, host_data):
    merged_data = {}

    merged_file = "{}_host_merged".format(os.path.splitext(database_data["profile"])[0])

    merged_data['profile'] = merged_file # Kmers profile
    df_classes = pd.DataFrame(database_data["classes"], columns = database_data["taxas"])
    if len(np.unique(df_classes['domain'])) != 1:
        df_classes[df_classes['domain'] != 'bacteria'] = 'bacteria'
    df_classes = df_classes.append(pd.DataFrame(host_data["classes"], columns = host_data["taxas"]), ignore_index = True)
    merged_data['classes'] = np.array(df_classes) # Class labels
    merged_data['kmers'] = database_data["kmers"] # Features
    merged_data['taxas'] = database_data["taxas"] # Known taxas for classification
    merged_data['fasta'] = (database_data['fasta'],host_data['fasta']) # Fasta fiule needed for reads simulation


    df_db = ray.data.read_parquet(database_data["profile"])
    df_host = ray.data.read_parquet(host_data["profile"])
    df_merged = df_db.union(df_host)
    df_merged.write_parquet(merged_file)

    return merged_data

# Unpack numpy tensor column to kmers columns
def unpack_kmers(df_file, lst_kmers):
    ray.data.set_progress_bars(False)
    df = ray.data.read_parquet(df_file)
    for i, col in enumerate(lst_kmers):
        df = df.add_column(col, lambda df: df['__value__'].to_numpy()[0][i])
    df = df.drop_columns(['__value__'])
    ray.data.set_progress_bars(True)
    return df