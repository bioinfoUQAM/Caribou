import os
import sys
import ray
import pickle
import cloudpickle

import pandas as pd

# Class construction
from abc import ABC, abstractmethod

__author__ = 'Nicolas de Montigny'

__all__ = ['ClassificationUtils']

class ClassificationUtils(ABC):
    """
    Utilities class for classifying sequences from metagenomes using ray

    ----------
    Attributes
    ----------

    ----------
    Methods
    ----------

    """
    def __init__(
        self,
        database_k_mers,
        k,
        outdirs,
        dataset,
        training_epochs,
        classifier,
        batch_size,
        verbose,
        cv
    ):
        # Parameters
        self.database_kmers = database_k_mers
        self.k = k
        self.outdirs = outdirs
        self.dataset = dataset
        self.training_epochs = training_epochs
        self.classifier = classifier
        self.batch_size = batch_size
        self.verbose = verbose
        self.cv = cv
        # Empty initializations
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None

    # Merge database and host data
    def merge_database_host(self, database_data, host_data):
        merged_data = {}

        merged_file = "{}_host_merged".format(os.path.splitext(database_data["profile"])[0])

        merged_data['profile'] = merged_file  # Kmers profile
        df_classes = pd.DataFrame(database_data["classes"], columns=database_data["taxas"])
        if len(np.unique(df_classes['domain'])) != 1:
            df_classes[df_classes['domain'] != 'bacteria'] = 'bacteria'
        df_classes = df_classes.append(pd.DataFrame(host_data["classes"], columns=host_data["taxas"]), ignore_index=True)
        merged_data['classes'] = np.array(df_classes)  # Class labels
        merged_data['kmers'] = database_data["kmers"]  # Features
        # Known taxas for classification
        merged_data['taxas'] = database_data["taxas"]
        # Fasta fiule needed for reads simulation
        merged_data['fasta'] = (database_data['fasta'], host_data['fasta'])

        df_db = ray.data.read_parquet(database_data["profile"])
        df_host = ray.data.read_parquet(host_data["profile"])
        df_merged = df_db.union(df_host)
        df_merged.write_parquet(merged_file)

        return merged_data

    # Unpack numpy tensor column to kmers columns
    def unpack_kmers(self, df_file, lst_kmers):
        ray.data.set_progress_bars(False)
        df = ray.data.read_parquet(df_file)
        for i, col in enumerate(lst_kmers):
            df = df.add_column(col, lambda df: df['__value__'].to_numpy()[0][i])
        df = df.drop_columns(['__value__'])
        ray.data.set_progress_bars(True)
        return df

    @abstractmethod
    def classify(self):
        """
        """