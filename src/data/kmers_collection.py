import os
import ray
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa

from glob import glob
from copy import copy
from shutil import rmtree
from subprocess import run
from joblib import Parallel, delayed, parallel_backend

from data.ray_tensor_lowvar_selection import TensorLowVarSelection

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
    """
    def __init__(
        self,
        seq_data,
        Xy_file,
        k,
        dataset,
        kmers_list = None,
        features_threshold = np.inf,
        nb_features_keep = np.inf
    ):
        ## Public attributes
        # Parameters
        self.k = k
        self.dataset = dataset
        self.Xy_file = Xy_file
        self.fasta = seq_data.data
        self._features_threshold = features_threshold
        self._nb_features_keep = nb_features_keep
        # Initialize empty
        self.df = None
        self.ids = []
        self.taxas = []
        self.classes = []
        self.method = None
        self.kmers_list = None
        self._labels = None
        self._lst_arr = []
        self._transformed = False
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
        self._files_list = []
        self._fasta_list = []

        ## Extraction
        # Execute
        self._compute_kmers()
        # Get informations from extracted data
        if self.kmers_list is None:
            self.kmers_list = list(self.kmers_list)
        if 'id' in self.kmers_list:
            self.kmers_list.remove('id')
        # Get labels that match K-mers extracted sequences
        if len(seq_data.labels) > 0:
            msk = np.array([True if id in self.ids else False for id in seq_data.ids])
            self.classes = seq_data.labels[msk]
        # Delete global tmp dir
        rmtree(self._tmp_dir)

    def _compute_kmers(self):
        self._get_fasta_list()
        # Extract kmers in parallel using KMC3
        self._parallel_extraction()
        # Build kmers matrix
        self._construct_data()

    def _get_fasta_list(self):
        if isinstance(self.fasta, list):
            self._fasta_list = self.fasta
        elif os.path.isfile(self.fasta):
            self._split_fasta()

    def _split_fasta(self):
        # Split files using faSplit
        cmd_split = f'{self._faSplit} byname {self.fasta} {self._tmp_dir}'
        os.system(cmd_split)
        # Get list of fasta files
        self._fasta_list = glob(os.path.join(self._tmp_dir, '*.fa'))

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
        tmp_folder = os.path.join(self._tmp_dir,f"tmp_{ind}")
        id = os.path.splitext(os.path.basename(file))[0]
        os.mkdir(tmp_folder)
        # Count k-mers with KMC
        cmd_count = os.path.join(self._kmc_path,f"kmc -k{self.k} -fm -ci10 -cs1000000000 -hp {file} {os.path.join(tmp_folder, str(ind))} {tmp_folder}")
        run(cmd_count, shell = True, capture_output=True)
        # Transform k-mers db with KMC
        cmd_transform = os.path.join(self._kmc_path,f"kmc_tools transform {os.path.join(tmp_folder, str(ind))} dump {os.path.join(self._tmp_dir, f'{ind}.txt')}")
        run(cmd_transform, shell = True, capture_output=True)
        # Transpose kmers profile
        profile = pd.read_table(os.path.join(self._tmp_dir,f"{ind}.txt"), sep = '\t', index_col = 0, header = None, names = ['id', str(id)]).T
        # Save seen kmers profile to parquet file
        if len(profile.columns) > 0:
            profile.reset_index(inplace=True)
            profile = profile.rename(columns = {'index':'id'})
            profile.to_csv(os.path.join(self._tmp_dir,f"{ind}.csv"), index = False)
        # Delete tmp dir and file
        rmtree(tmp_folder)
        os.remove(os.path.join(self._tmp_dir,f"{ind}.txt"))
        return list(profile.columns)

    def _extract_given_kmers(self, ind, file, kmers_list):
        id = None
        arr = []
        # Make tmp folder per sequence
        tmp_folder = os.path.join(self._tmp_dir,f"tmp_{ind}")
        id = os.path.splitext(os.path.basename(file))[0]
        os.mkdir(tmp_folder)
        # Count k-mers with KMC
        cmd_count = os.path.join(self._kmc_path,f"kmc -k{self.k} -fm -cs1000000000 -hp {file} {os.path.join(tmp_folder, str(ind))} {tmp_folder}")
        run(cmd_count, shell = True, capture_output=True)
        # Transform k-mers db with KMC
        cmd_transform = os.path.join(self._kmc_path, f"kmc_tools transform {os.path.join(tmp_folder, str(ind))} dump {os.path.join(self._tmp_dir,f'{ind}.txt')}")
        run(cmd_transform, shell = True, capture_output=True)
        # Transpose kmers profile
        seen_profile = pd.read_table(os.path.join(self._tmp_dir,f"{ind}.txt"), sep = '\t', index_col = 0, header = None, names = ['id', str(id)]).T
        # List of seen kmers
        seen_kmers = list(seen_profile.columns)
        arr = np.zeros((1,len(kmers_list)))
        if len(seen_kmers) > 0:
            id = seen_profile.index[0]
            for col in seen_kmers:
                if col in kmers_list:
                    arr[0, kmers_list.index(col)] = seen_profile.at[id, col]
        # Delete tmp dir and file
        rmtree(tmp_folder)
        os.remove(os.path.join(self._tmp_dir, f"{ind}.txt"))
        return (id, ray.put(arr))

    def _construct_data(self):
        if self.method == 'seen':
            self._files_list = glob(os.path.join(self._tmp_dir,'*.csv')) # List csv files
            # Read/concatenate files csv -> memory tensors -> Ray
            self._batch_read_write_seen()
        elif self.method == 'given':
            # Read/concatenate memory tensors -> Ray
            self._batch_read_write_given()
        # Delete tmp tensors in memory
        for ref in self._lst_arr:
            del ref

    # Map csv files to numpy array refs then write to parquet file with Ray
    def _batch_read_write_seen(self):
        print('_batch_read_write_seen')
        for file in self._files_list:
            tmp = pd.read_csv(file)
            self.ids.append(tmp.loc[0,'id'])
            arr = np.zeros((1, len(self.kmers_list)-1))
            cols = list(tmp.columns)
            cols.remove('id')
            for col in cols:
                arr[0, self.kmers_list.index(col)] = tmp.at[0, col]
            self._lst_arr.append(ray.put(arr))

        self.df = ray.data.from_numpy_refs(self._lst_arr)

        # Diminish nb of features
        feature_selector = TensorLowVarSelection(
            '__value__',
            self.kmers_list,
            threshold = self._features_threshold,
            nb_keep = self._nb_features_keep,
        )
        self.df = feature_selector.fit_transform(self.df)
        self._transformed = False if feature_selector.transform_stats() is None else True
        if self._transformed:
            self.kmers_list = [kmer for kmer in self.kmers_list if kmer not in feature_selector.removed_features]

        self._zip_id_col()
        self.df.write_parquet(self.Xy_file)

    def _batch_read_write_given(self):
        print('_batch_read_write_given')
        self.df = ray.data.from_numpy_refs(self._lst_arr)
        self._zip_id_col()
        self.df.write_parquet(self.Xy_file)
    
    def _zip_id_col(self):
        num_blocks = self.df.num_blocks()
        len_df = self.df.count()
        ids = pd.DataFrame({
            'id': self.ids
        })
        self._ensure_length_ds(len_df, len(ids))
        # Convert ids -> ray.data.Dataset with arrow schema
        if self._transformed:
            ids = ray.data.from_pandas(ids)
        else:
            ids = ray.data.from_arrow(pa.Table.from_pandas(ids))
        # Repartition to 1 row/partition
        self.df = self.df.repartition(len_df)
        ids = ids.repartition(len_df)
        # Ensure both ds fully executed
        for ds in [self.df, ids]:
            if not ds.is_fully_executed():
                ds.fully_executed()
        # Zip X and y
        self.df = self.df.zip(ids).repartition(num_blocks)

    def _ensure_length_ds(self, len_df, len_ids):
        if len_df != len_ids:
            raise ValueError(
                f'K-mers profile and ids list have different lengths: {len_df} and {len_ids}')
