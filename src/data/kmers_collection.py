import os
import gc
import ray
import warnings

import numpy as np
import pandas as pd

from glob import glob
from shutil import rmtree
from subprocess import run
from itertools import chain
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
        self._nb_kmers = 0
        self._index = {}
        self._labels = None
        self._transformed = False
        # Get labels from seq_data
        if len(seq_data.labels) > 0:
            self._labels = pd.DataFrame(seq_data.labels, columns = seq_data.taxas, index = seq_data.ids)
            self._labels = self._labels.reset_index(names = ['id'])
        # Get taxas from seq_data if not empty
        if len(seq_data.taxas) > 0:
            self.taxas = seq_data.taxas
        # Infer method from presence of already extracted kmers or not
        if isinstance(kmers_list, list):
            self.method = 'given'
            self.kmers_list = kmers_list
            self._nb_kmers = len(self.kmers_list)
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
        # Delete global tmp dir
        rmtree(self._tmp_dir)

    def _compute_kmers(self):
        self._get_fasta_list()
        # Extract kmers in parallel using KMC3
        self._kmers_extraction()
        # Build kmers matrix
        self._build_dataset()

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

    def _kmers_extraction(self):
        print('extract_kmers in parallel')
        # Only extract k-mers using KMC in parallel
        with parallel_backend('threading'):
            print('_parallel_KMC_kmers')
            Parallel(n_jobs = -1, prefer = 'threads', verbose = 1)(
                delayed(self._parallel_KMC_kmers)
                (file) for file in self._fasta_list)
        # List profile files
        self._files_list = glob(os.path.join(self._tmp_dir,'*.txt'))

        if self.method == 'seen':
            self._reduce_features()

    def _parallel_KMC_kmers(self, file):
        # Make tmp folder per sequence
        id = os.path.splitext(os.path.basename(file))[0]
        tmp_folder = os.path.join(self._tmp_dir,f"tmp_{id}")
        os.mkdir(tmp_folder)
        # Count k-mers with KMC
        cmd_count = os.path.join(self._kmc_path,f"kmc -k{self.k} -fm -ci0 -cs1000000000 -hp {file} {os.path.join(tmp_folder, id)} {tmp_folder}")
        run(cmd_count, shell = True, capture_output=True)
        # Transform k-mers db with KMC
        cmd_transform = os.path.join(self._kmc_path,f"kmc_tools transform {os.path.join(tmp_folder, id)} dump {os.path.join(self._tmp_dir, f'{id}.txt')}")
        run(cmd_transform, shell = True, capture_output=True)
        rmtree(tmp_folder)

    # Diminish nb of features
    def _reduce_features(self):
        print('_reduce_features')
        with parallel_backend('threading'):
            print('parallel _get_kmers_list')
            lst_col = Parallel(n_jobs = -1, prefer = 'threads', verbose = 1)(
                delayed(self._get_kmers_list)
                (file) for file in self._files_list)
        self._flatten_kmers_array(lst_col)
        self._kmers_indexing()
        index_var = self._index_variation()
        self._features_choice(index_var)

    # Parallel extract k-mers from profiles
    def _get_kmers_list(self, file):
        lst_col = []
        id = os.path.splitext(os.path.basename(file))[0]
        profile = pd.read_table(file, sep = '\t', index_col = 0, header = None, names = ['id', str(id)])
        if len(profile.index) > 0:
            lst_col = profile.index
        return np.array(lst_col)

    # Flatten list of arrays
    def _flatten_kmers_array(self, lst_col):
        self.kmers_list = list(set(chain.from_iterable(lst_col)))
        self._nb_kmers = len(self.kmers_list)

    # Index k-mers abundances
    def _kmers_indexing(self):
        print('_kmers_index')
        for kmer in self.kmers_list:
            self._index[kmer] = np.zeros(len(self._files_list))
        for i, file in enumerate(self._files_list):
            id = os.path.splitext(os.path.basename(file))[0]
            profile = pd.read_table(file, sep = '\t', index_col = 0, header = None, names = ['id', str(id)])
            for kmer in self.kmers_list:
                if kmer in profile.index:
                    self._index[kmer][i] = profile.loc[kmer,id]
    
    # Compute variance for each k-mers indexed
    def _index_variation(self):
        print('_index_variation')
        index_var = {}
        for kmer in self.kmers_list:
            arr = self._index[kmer]
            index_var[kmer] = np.var(arr)
        return index_var
    
    # Exclude k-mers features when variance is lower than 25% or higher than 75% of variance distribution
    def _features_choice(self, index_var):
        print('_features_choice')
        s_var = pd.Series(index_var)
        quartiles = s_var.quantile([0.25,0.75])
        s_var = s_var[s_var>quartiles[0.25]]
        s_var = s_var[s_var<quartiles[0.75]] # 867
        self.kmers_list = list(s_var.index)
        self._nb_kmers = len(self.kmers_list)

    # Map csv files to numpy array then write to parquet file with ray
    def _build_dataset(self):
        print('_construct_dataset')
        with parallel_backend('threading'):
            print('parallel _get_kmers_list')
            Parallel(n_jobs = -1, prefer = 'threads', verbose = 1)(
                delayed(self._map_profile_to_tensor)
                (file) for file in self._files_list)
        lst_tensor_files = glob(os.path.join(self._tmp_dir,'*.parquet'))
        self._convert_tensors_ray_ds(lst_tensor_files)
        # if self.method == 'seen' and (self._features_threshold != np.inf or self._nb_features_keep != np.inf) :
        #     self._reduce_features()

    def _read_kmers_profile(self, file):
        id = os.path.splitext(os.path.basename(file))[0]
        profile = pd.read_table(file, sep = '\t', index_col = 0, header = None, names = ['id', id])
        profile_kmers = profile.index
        return(profile, profile_kmers, id)

    def _map_profile_to_tensor(self, file):
        profile, profile_kmers, id = self._read_kmers_profile(file)
        tensor = np.zeros((1, self._nb_kmers))
        for kmer in profile_kmers:
            if kmer in self.kmers_list:
                tensor[0, self.kmers_list.index(kmer)] = profile.at[kmer, id]
        tensor = ray.data.from_numpy(tensor)
        tensor = tensor.add_column('id', lambda x : id)
        tensor.write_parquet(self._tmp_dir)

    def _convert_tensors_ray_ds(self, lst_arr):
        print('_convert_tensors_ray_ds')
        self.df = ray.data.read_parquet_bulk(lst_arr)
        for row in self.df.iter_rows():
            self.ids.append(row['id'])
        if self._labels is not None:
            self._get_classes_labels()
        self.df.write_parquet(self.Xy_file)

    def _get_classes_labels(self):
        print('_get_classes_labels')
        self.classes = pd.DataFrame({'id' : self.ids})
        self.classes = self.classes.merge(self._labels, on = 'id', how = 'left')
        self.classes = self.classes.drop('id', axis = 1)
        self.classes = np.array(self.classes)
