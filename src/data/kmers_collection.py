import os
import ray
import warnings

import numpy as np
import modin.pandas as pd

from glob import glob
from shutil import rmtree
from subprocess import run
from joblib import Parallel, delayed, parallel_backend

__author__ = ['Amine Remita', 'Nicolas de Montigny']

__all__ = ['KmersCollection','build_kmers_Xy_data','build_kmers_X_data']

"""
Module adapted from module kmer_collections.py of
mlr_kgenomvir package [Remita et al. 2022]

Save kmers profiles directly to drive instead of memory and adapted / added functions to do so.
Using Ray datasets for I/O and to scale cluster to available computing ressources.
"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

class KmersCollection():
    """
    ----------
    Attributes
    ----------

    k : int
        The length of K-mers extracted

    dataset : string
        Name of the dataset from which the K-mers profiles were extracted

    Xy_file : string
        Path to a folder containing the Ray Dataset of K-mers abundance profiles
        The folder contains a number of files in Apache parquet format
        The number of files is equivalent to the number of blocks in the dataset

    fasta : string
        A fasta file containing all sequences from which K-mers were extracted

    df : ray.data.Dataset
        A Ray dataset containing the K-mers abundance profiles of each sequences

    taxas : list of strings
        A list containing the taxas contained in the dataset if they were present
        Returns None if no taxas were present in the dataset

    classes : list of strings
        A list containing the classes contained in the dataset if they were present
        It must be paired to the attribute 'taxas' to be used
        Returns None if no classes were present in the dataset

    method : string
        Method used to extract K-mers :
            'given' if a K-mers list was passed in parameters
            'seen' if no K-mers list was passed in parameters

    kmers_list : list of strings
        List of given K-mers if one was passed in parameters
        List of K-mers extracted if none was passed in parameters

    """
    def __init__(self, seq_data, Xy_file, k, dataset, kmers_list = None):
        ## Public attributes
        # Parameters
        self.k = k
        self.dataset = dataset
        self.Xy_file = Xy_file
        self.fasta = seq_data.data
        # Initialize empty
        self.df = None
        self.taxas = []
        self.classes = []
        self.method = None
        self.kmers_list = None
        # Get taxas from seq_data if not empty
        if len(seq_data.taxas) > 0:
            self.taxas = seq_data.taxas
        # Infer method from presence of already extracted kmers or not
        if isinstance(kmers_list, list):
            self.method = 'given'
            self.kmers_list = kmers_list
        else:
            self.method = 'seen'

        ## Internal attributes
        # Global tmp dir path
        self._tmp_dir = os.path.join(os.path.split(Xy_file)[0],"tmp","")
        # Make global tmp dir if it doesn't exist
        if not os.path.isdir(self._tmp_dir):
            os.mkdir(self._tmp_dir)
        # Path to third-party utilities
        self._kmc_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"KMC","bin")
        self._faSplit = os.path.join(os.path.dirname(os.path.realpath(__file__)),"faSplit")
        # Initialize empty
        self._csv_list = None
        self._fasta_list = None

        ## Extraction
        # Execute
        self._compute_kmers()
        # Get informations from extracted data
        if self.kmers_list is None:
            self.kmers_list = list(self.df.limit(1).to_pandas().columns)
            self.kmers_list.remove('id')
        # Get labels that match K-mers extracted sequences
        if len(seq_data.labels) > 0:
            ids = list(self.df.to_modin()['id'])
            msk = np.array([True if id in ids else False for id in seq_data.ids])
            self.classes = seq_data.labels[msk]
        # Delete global tmp dir
        rmtree(self._tmp_dir)

    def _compute_kmers(self):
        # Split files using faSplit
        cmd_split = '{} byname {} {}'.format(self._faSplit, self.fasta, self._tmp_dir)
        os.system(cmd_split)
        # Get list of fasta files
        self._fasta_list = glob(os.path.join(self._tmp_dir,'*.fa'))
        # Extract kmers in parallel using KMC3
        self._parallel_extraction()
        # Delete tmp fasta files
        for file in self._fasta_list:
            os.remove(file)
        # build kmers matrix
        self._construct_data()

    def _parallel_extraction(self):
        if self.method == 'seen':
            with parallel_backend('threading'):
                Parallel(n_jobs = -1, prefer = 'threads', verbose = 100)(
                delayed(self._extract_seen_kmers)
                (i, file) for i, file in enumerate(self._fasta_list))
        elif self.method == 'given':
            with parallel_backend('threading'):
                Parallel(n_jobs = -1, prefer = 'threads', verbose = 100)(
                delayed(self._extract_given_kmers)
                (i, file) for i, file in enumerate(self._fasta_list))

    def _extract_seen_kmers(self, ind, file):
        # Make tmp folder per sequence
        tmp_folder = os.path.join(self._tmp_dir,"tmp_{}".format(ind))
        id = os.path.splitext(os.path.basename(file))[0]
        os.mkdir(tmp_folder)
        # Count k-mers with KMC
        cmd_count = os.path.join(self._kmc_path,"kmc -k{} -fm -ci5 -cs1000000000 -m10 -hp {} {} {}".format(self.k, file, os.path.join(tmp_folder, str(ind)), tmp_folder))
        run(cmd_count, shell = True, capture_output=True)
        # Transform k-mers db with KMC
        cmd_transform = os.path.join(self._kmc_path,"kmc_tools transform {} dump {}".format(os.path.join(tmp_folder, str(ind)), os.path.join(self._tmp_dir, "{}.txt".format(ind))))
        run(cmd_transform, shell = True, capture_output=True)
        # Transpose kmers profile
        profile = pd.read_table(os.path.join(self._tmp_dir,"{}.txt".format(ind)), sep = '\t', header = None, names = ['id', str(id)]).T
        # Save seen kmers profile to csv file
        if len(profile.columns) > 1:
            profile.to_csv(os.path.join(self._tmp_dir,"{}.csv".format(ind)), header = False)
        # Delete tmp dir and file
        rmtree(tmp_folder)
        os.remove(os.path.join(self._tmp_dir,"{}.txt".format(ind)))

    def _extract_given_kmers(self, ind, file):
        # Make tmp folder per sequence
        tmp_folder = os.path.join(self._tmp_dir,"tmp_{}".format(ind))
        id = os.path.splitext(os.path.basename(file))[0]
        os.mkdir(tmp_folder)
        # Count k-mers with KMC
        cmd_count = os.path.join(self._kmc_path,"kmc -k{} -fm -ci4 -cs1000000000 -m10 -hp {} {} {}".format(self.k, file, os.path.join(tmp_folder, str(ind)), tmp_folder))
        run(cmd_count, shell = True, capture_output=True)
        # Transform k-mers db with KMC
        cmd_transform = os.path.join(self._kmc_path,"kmc_tools transform {} dump {}".format(os.path.join(tmp_folder, str(ind)), os.path.join(self._tmp_dir, "{}.txt".format(ind))))
        run(cmd_transform, shell = True, capture_output=True)
        # Transpose kmers profile
        seen_profile = pd.read_table(os.path.join(self._tmp_dir,"{}.txt".format(ind)), sep = '\t', header = None, names = ['id', str(id)]).T
        # List of seen kmers
        seen_kmers = list(seen_profile.columns)
        seen_kmers.remove('id')
        if len(seen_kmers) > 1:
            # Tmp df to write given kmers to file
            given_profile = pd.DataFrame(np.zeros((1,len(self.kmers_list))), columns = self.kmers_list, index = [id])
            # Keep only given kmers that were found
            for kmer in self.kmers_list:
                if kmer in seen_kmers:
                    given_profile.at[id,kmer] = seen_profile.loc[id,kmer]
                else:
                    given_profile.at[id,kmer] = 0
            # Save given kmers profile to csv file
            given_profile.to_csv(os.path.join(self._tmp_dir,"{}.csv".format(ind)), header = False, index_label = 'id')
        # Delete temp dir and file
        rmtree(tmp_folder)
        os.remove(os.path.join(self._tmp_dir,"{}.txt".format(ind)))

    def _construct_data(self):
        self._csv_list = glob(os.path.join(self._tmp_dir,'*.csv'))
        # Read/concatenate files with Ray by batches
        nb_batch = 0
        while np.ceil(len(self._csv_list)/1000) > 1:
            batches_list = np.array_split(self._csv_list, np.ceil(len(self._csv_list)/1000))
            batch_dir = os.path.join(self._tmp_dir, 'batch_{}'.format(nb_batch))
            os.mkdir(batch_dir)
            for batch in batches_list:
                self._batch_read_write(list(batch), batch_dir)
            self._csv_list = glob(os.path.join(batch_dir,'*.csv'))
            nb_batch += 1
        # Read/concatenate batches with Ray
        self.df = ray.data.read_csv(self._csv_list)
        # Fill NAs with 0
        self.df.map_batches(self._na_2_zero, batch_format = 'pandas')
        # Save dataset
        self.df.write_parquet(self.Xy_file)

    def _batch_read_write(self, batch, dir):
        df = ray.data.read_csv(batch)
        df.write_csv(dir)
        for file in batch:
            os.remove(file)

    def _na_2_zero(self, df):
        df = df.fillna(0)
        cols = list(df.columns)
        cols.remove('id')
        df[cols] = df[cols].astype(np.int32)
        print(df)
        return df
