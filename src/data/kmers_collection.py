import os
import ray
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from glob import glob
from copy import copy
from shutil import rmtree
from subprocess import run
from joblib import Parallel, delayed, parallel_backend

__author__ = ['Amine Remita', 'Nicolas de Montigny']

__all__ = ['KmersCollection']

"""
Module adapted from module kmer_collections.py of
mlr_kgenomvir package [Remita et al. 2022]

Save kmers profiles into tensors then directly to drive and adapted / added functions to do so.
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

    ids : list
        A list of all sequences ids

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

    ----------
    Methods
    ----------

    unpack_kmers()

    """
    def __init__(
        self,
        seq_data,
        Xy_file,
        k,
        dataset,
        kmers_list = None
    ):
        ## Public attributes
        # Parameters
        self.k = k
        self.dataset = dataset
        self.Xy_file = Xy_file
        self.fasta = seq_data.data
        # Initialize empty
        self.df = None
        self.ids = []
        self.taxas = []
        self.classes = []
        self.method = None
        self.kmers_list = None
        self._labels = None
        self._lst_arr = []
        # Get labels from seq_data
        if len(seq_data.labels) > 0:
            self._labels = pd.DataFrame(seq_data.labels, columns = seq_data.taxas, index = seq_data.ids)
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
        self._files_list = None
        self._fasta_list = None

        ## Extraction
        # Execute
        self._compute_kmers()
        # Get informations from extracted data
        if self.kmers_list is None:
            self.kmers_list = list(self.kmers_list)
        # Get labels that match K-mers extracted sequences
        if len(seq_data.labels) > 0:
            msk = np.array([True if id in self.ids else False for id in seq_data.ids])
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
            print('seen_kmers')
            lst_col = []
            with parallel_backend('threading'):
                lst_col = Parallel(n_jobs = -1, prefer = 'threads', verbose = 1)(
                    delayed(self._extract_seen_kmers)
                    (i, file) for i, file in enumerate(self._fasta_list))
            # Get list of all columns in files in parallel
            self.kmers_list = list(np.unique(np.concatenate(lst_col)))
        elif self.method == 'given':
            print('given_kmers')
            lst_ids_arr = []
            with parallel_backend('threading'):
                lst_ids_arr =  Parallel(n_jobs = -1, prefer = 'threads', verbose = 1)(
                    delayed(self._extract_given_kmers)
                    (i, file, copy(self.kmers_list)) for i, file in enumerate(self._fasta_list))
            for id, arr in lst_ids_arr:
                self.ids.append(id)
                self._lst_arr.append(arr)

    def _extract_seen_kmers(self, ind, file):
        # Make tmp folder per sequence
        tmp_folder = os.path.join(self._tmp_dir,"tmp_{}".format(ind))
        id = os.path.splitext(os.path.basename(file))[0]
        os.mkdir(tmp_folder)
        # Count k-mers with KMC
        cmd_count = os.path.join(self._kmc_path,"kmc -k{} -fm -ci10 -cs1000000000 -hp {} {} {}".format(self.k, file, os.path.join(tmp_folder, str(ind)), tmp_folder))
        run(cmd_count, shell = True, capture_output=True)
        # Transform k-mers db with KMC
        cmd_transform = os.path.join(self._kmc_path,"kmc_tools transform {} dump {}".format(os.path.join(tmp_folder, str(ind)), os.path.join(self._tmp_dir, "{}.txt".format(ind))))
        run(cmd_transform, shell = True, capture_output=True)
        # Transpose kmers profile
        profile = pd.read_table(os.path.join(self._tmp_dir,"{}.txt".format(ind)), sep = '\t', index_col = 0, header = None, names = ['id', str(id)]).T
        # Save seen kmers profile to parquet file
        if len(profile.columns) > 0:
            profile.reset_index(inplace=True)
            profile = profile.rename(columns = {'index':'id'})
            profile.to_csv(os.path.join(self._tmp_dir,"{}.csv".format(ind)), index = False)
        # Delete tmp dir and file
        rmtree(tmp_folder)
        os.remove(os.path.join(self._tmp_dir,"{}.txt".format(ind)))
        return list(profile.columns)

    def _extract_given_kmers(self, ind, file, kmers_list):
        id = None
        arr = []
        # Make tmp folder per sequence
        tmp_folder = os.path.join(self._tmp_dir,"tmp_{}".format(ind))
        id = os.path.splitext(os.path.basename(file))[0]
        os.mkdir(tmp_folder)
        # Count k-mers with KMC
        cmd_count = os.path.join(self._kmc_path,"kmc -k{} -fm -cs1000000000 -hp {} {} {}".format(self.k, file, os.path.join(tmp_folder, str(ind)), tmp_folder))
        run(cmd_count, shell = True, capture_output=True)
        # Transform k-mers db with KMC
        cmd_transform = os.path.join(self._kmc_path,"kmc_tools transform {} dump {}".format(os.path.join(tmp_folder, str(ind)), os.path.join(self._tmp_dir, "{}.txt".format(ind))))
        run(cmd_transform, shell = True, capture_output=True)
        # Transpose kmers profile
        seen_profile = pd.read_table(os.path.join(self._tmp_dir,"{}.txt".format(ind)), sep = '\t', index_col = 0, header = None, names = ['id', str(id)]).T
        # List of seen kmers
        seen_kmers = list(seen_profile.columns)
        if len(seen_kmers) > 0:
            id = seen_profile.index[0]
            arr = np.zeros((1,len(kmers_list)))
            for col in seen_kmers:
                arr[0, kmers_list.index(col)] = seen_profile.at[id, col]
        # Delete tmp dir and file
        rmtree(tmp_folder)
        os.remove(os.path.join(self._tmp_dir, "{}.txt".format(ind)))
        return (id, ray.put(arr))

    def _construct_data(self):
        if self.method == 'seen':
            self._files_list = glob(os.path.join(self._tmp_dir,'*.csv')) # List csv files
            # Read/concatenate files csv -> memory tensors -> Ray
            self._batch_read_write_seen()
        elif self.method == 'given':
            # Read/concatenate memory tensors -> Ray
            self._batch_read_write_given()

    # Map csv files to numpy array refs then write to parquet file with Ray
    def _batch_read_write_seen(self):
        for file in self._files_list:
            tmp = pd.read_csv(file)
            self.ids.append(tmp.loc[0,'id'])
            arr = np.zeros((1, len(self.kmers_list)-1))
            cols = list(tmp.columns)
            cols.remove('id')
            for col in cols:
                    arr[0, self.kmers_list.index(col)] = tmp.at[0, col]            
            self._lst_arr.append(ray.put(arr))
            os.remove(file)
        self.df = ray.data.from_numpy_refs(self._lst_arr)
        self.df = self.df.add_column('id', lambda ds : pd.DataFrame(self.ids))
        self.df.write_parquet(self.Xy_file)

    def _batch_read_write_given(self):
        self.df = ray.data.from_numpy_refs(self._lst_arr)
        self.df = self.df.add_column('id', lambda ds : pd.DataFrame(self.ids))
        self.df.write_parquet(self.Xy_file)

    # Unpack numpy tensor column to kmers columns
    def unpack_kmers(self):
        if not os.path.isdir(self.Xy_file):
            raise ValueError("K-mers were not extracted yet. Please instantiate the object first.")
        if self.df is None:
            self.df = ray.data.read_parquet(self.Xy_file)
        ray.data.set_progress_bars(False)
        for i, col in enumerate(self.kmers_list):
            self.df = self.df.add_column(col, lambda df: df['__value__'].to_numpy()[0][i])
        self.df = self.df.drop_columns(['__value__'])
        ray.data.set_progress_bars(True)