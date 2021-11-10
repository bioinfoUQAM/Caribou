
import pandas as pd
import numpy as np
import tables as tb

import os

from data.generators import DataGenerator

__author__ = "nicolas"

# Load data from file
def load_Xy_data(Xy_file):
    if os.path.basename(Xy_file).split(sep = ".")[1] == "npz":
        with np.load(Xy_file, allow_pickle=True) as f:
            return f['data'].tolist()

# Save data to file
def save_Xy_data(data, Xy_file):
    if type(data) == pd.core.frame.DataFrame:
        with tb.open_file(Xy_file, "a") as handle:
            array = handle.create_carray("/", "data", obj = np.array(data,dtype=np.float32))
    elif type(data) == dict:
        np.savez(Xy_file, data=data)

def save_predicted_kmers(positions_list, y, kmers_list, ids, infile, outfile):
    data = False
    generator = DataGenerator(infile, y, 1, kmers_list, ids, cv = 0, shuffle = False)
    with tb.open_file(outfile, "a") as handle:
        for i, (X, y) in enumerate(generator.iterator):
            if i in positions_list and not data:
                data = handle.create_earray("/", "data", obj = np.array(X))
            elif i in positions_list and data:
                data.append(np.array(X))
    generator.handle.close()

def merge_database_host(database_data, host_data):
    merged_data = dict()

    path, ext = os.path.splitext(database_data["X"])
    merged_file = "{}_host_merged{}".format(path, ext)

    merged_data["X"] = merged_file
    merged_data["y"] = np.concatenate((database_data["y"], host_data["y"]))
    merged_data["ids"] = database_data["ids"] + host_data["ids"]
    merged_data["kmers_list"] = list(set(database_data["kmers_list"]).union(host_data["kmers_list"]))
    merged_data["taxas"] = max(database_data["taxas"], host_data["taxas"], key = len)

    generator_database = DataGenerator(database_data["X"], database_data["y"], 32, database_data["kmers_list"], database_data["ids"], cv = 0, shuffle = False)
    generator_host = DataGenerator(host_data["X"], host_data["y"], 32, host_data["kmers_list"], host_data["ids"], cv = 0, shuffle = False)
    if not os.path.isfile(merged_file):
        data = False
        with tb.open_file(merged_file, "a") as handle:
            for (X_d, y_d), (X_h, y_h) in zip(generator_database.iterator, generator_host.iterator):
                if not data:
                    data = handle.create_earray("/", "data", obj = np.array(pd.merge(X_d, X_h, how = "outer")))
                else:
                    data.append(np.array(pd.merge(X_d, X_h, how = "outer")))
        generator_database.handle.close()
        generator_host.handle.close()

    return merged_data
