
import modin.pandas as pd
import numpy as np

import os
import ray
import joblib

__author__ = "Nicolas de Montigny"

__all__ = ['load_Xy_data','save_Xy_data','merge_database_host']

# Load data from file
def load_Xy_data(Xy_file):
    with np.load(Xy_file, allow_pickle=True) as f:
        return f['data'].tolist()

# Save data to file
def save_Xy_data(df, Xy_file):
    np.savez(Xy_file, data = df)

def merge_database_host(database_data, host_data):
    merged_data = {}

    merged_file = "{}_host_merged".format(os.path.splitext(database_data["profile"])[0])

    merged_data['profile'] = merged_file # Kmers profile
    merged_data['classes'] = np.array(pd.DataFrame(database_data["classes"], columns = database_data["taxas"]).append(pd.DataFrame(host_data["classes"], columns = host_data["taxas"]), ignore_index = True)) # Class labels
    merged_data['kmers'] = database_data["kmers"] # Features
    merged_data['taxas'] = database_data["taxas"] # Known taxas for classification

    df_db = ray.data.read_parquet(database_data["profile"])
    df_host = ray.data.read_parquet(host_data["profile"])
    df_merged = df_db.union(df_host)
    df_merged.write_parquet(merged_file)

    return merged_data
