
import modin.pandas as pd
import numpy as np

import os
import ray
import joblib

__author__ = "Nicolas de Montigny"

__all__ = ['load_Xy_data','save_Xy_data','save_predicted_kmers','merge_database_host','label_encode']

# Load data from file
def load_Xy_data(Xy_file):
    with np.load(Xy_file, allow_pickle=True) as f:
        return f['data'].tolist()

# Save data to file
def save_Xy_data(df, Xy_file):
    np.savez(Xy_file, data = df)

def merge_database_host(database_data, host_data):
    merged_data = dict()

    path, ext = os.path.splitext(database_data["X"])
    merged_file = "{}_host_merged{}".format(path, ext)

    merged_data["X"] = merged_file
    merged_data["y"] = np.array(pd.DataFrame(database_data["y"], columns = database_data["taxas"]).append(pd.DataFrame(host_data["y"], columns = host_data["taxas"]), ignore_index = True))
    merged_data["ids"] = database_data["ids"] + host_data["ids"]
    merged_data["kmers_list"] = database_data["kmers_list"]
    merged_data["taxas"] = list(set(database_data["taxas"]).union(host_data["taxas"]))

    df_db = pd.read_csv(database_data["X"])
    df_host = pd.read_csv(host_data["X"])
    df_merged = pd.concat([df_db, df_host])
    df_merged.to_csv(merged_file)

    return merged_data
