
import pandas as pd
import numpy as np
import tables as tb

import os

from tensorflow.keras.utils import to_categorical

from Caribou.data.generators import iter_generator

__author__ = "Nicolas de Montigny"

__all__ = ['load_Xy_data','save_Xy_data','save_predicted_kmers','merge_database_host','to_int_cls']

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

def save_predicted_kmers(positions_list, y, kmers_list, ids, infile, outfile, classif):
    data = None
    if classif == "binary":
        generator = iter_generator(infile, y, 1, kmers_list, ids, classif, cv = 0, shuffle = False, training = False, positions_list = positions_list)
        with tb.open_file(outfile, "w") as handle:
            for X, y in generator.iterator:
                df = np.array(X, dtype = "float32")
                if data is None:
                    data = handle.create_earray("/", "data", obj = df)
                else:
                    data.append(df)
        generator.handle.close()
    elif classif == "multi":
        generator = iter_generator(infile, y, 1, kmers_list, ids, classif, cv = 0, shuffle = False, training = False, positions_list = positions_list)
        for i, (X, y) in enumerate(generator.iterator):
            with pd.HDFStore(outfile) as data:
                X.fillna(0, axis = 0, inplace = True)
                X = X.astype('int64')
                if not os.path.isfile(outfile):
                    X.to_hdf(data, "data", format = "table", mode = "w", min_itemsize = len(max(ids, key = len)))
                else:
                    X.to_hdf(data, "data", format = "table", mode = "a", append = True)
        generator.handle.close()

def merge_database_host(database_data, host_data):
    merged_data = dict()

    path, ext = os.path.splitext(database_data["X"])
    merged_file = "{}_host_merged{}".format(path, ext)

    merged_data["X"] = merged_file
    merged_data["y"] = np.array(pd.merge(pd.DataFrame(database_data["y"], columns = database_data["taxas"]), pd.DataFrame(host_data["y"], columns = host_data["taxas"]), how = "outer"))
    merged_data["ids"] = database_data["ids"] + host_data["ids"]
    merged_data["kmers_list"] = database_data["kmers_list"]
    merged_data["taxas"] = list(set(database_data["taxas"]).union(host_data["taxas"]))

    generator_database = iter_generator(database_data["X"], database_data["y"], 1, database_data["kmers_list"], database_data["ids"], np.arange(len(database_data["ids"])), cv = 0, shuffle = False)
    generator_host = iter_generator(host_data["X"], host_data["y"], 1, host_data["kmers_list"], host_data["ids"], np.arange(len(host_data["ids"])), cv = 0, shuffle = False)
    if not os.path.isfile(merged_file):
        data = False
        with tb.open_file(merged_file, "a") as handle:
            for X_d, y_d in generator_database.iterator:
                if not data:
                    data = handle.create_earray("/", "data", obj = np.array(X_d, dtype = np.uint64))
                else:
                    data.append(np.array(X_d, dtype = np.uint64))
            for X_h, y_h in generator_host.iterator:
                data.append(np.array(X_h, dtype = np.uint64))
        generator_database.handle.close()
        generator_host.handle.close()

    return merged_data

def to_int_cls(data, nb_cls, list_cls):
    # integer encoding of labels
    for cls in list_cls:
        pos_cls = np.where(list_cls == cls)[0]
        data.replace(to_replace = {cls : pos_cls}, inplace = True)
    return data