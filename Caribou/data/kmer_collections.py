from Caribou.data.seq_collections import SeqCollection

import re
import os
import gzip

from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from Bio import SeqIO
from os.path import splitext
from subprocess import run
from shutil import rmtree
from joblib import Parallel, delayed, parallel_backend
from dask.distributed import Client

import numpy as np
import pandas as pd
import tables as tb
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.feature_selection import VarianceThreshold

# From mlr_kgenomvir
__author__ = ['Amine Remita', 'Nicolas de Montigny']

__all__ = [ 'SeenKmersCollection','GivenKmersCollection','build_kmers','build_kmers_Xy_data','build_kmers_X_data']

"""
Module adapted from module kmer_collections.py of
mlr_kgenomvir package [Remita et al. 2021]

Save kmers directly to drive instead of memory and
adapted / added functions to do so.
"""

# #####
# Helper functions
# ################

def get_index_from_kmer(kmer, k):
    """
    Function adapted from module enrich.pyx of
    GenomeClassifier package [Sandberg et al. (2001)]

    Instead of starting by f=1 and multiplying it by 4,
    it starts with f= 4**(k-1) and divide it by 4
    in each iteration

    The returned index respects the result of itertools.product()
    ["".join(t) for t in itertools.product('ACGT', repeat=k)]
    """

    f= 4 ** (k-1)
    s=0
    alpha_to_code = {'A':0, 'C':1, 'G':2, 'T':3}

    for i in range(0, k):
        alpha_code=alpha_to_code[kmer[i]]
        s = s + alpha_code * f
        f = f // 4

    return s


# #####
# Kmers collections
# ##################

class KmersCollection(ABC):

    def __compute_kmers_from_collection(self, sequences):
        for i, seq in enumerate(sequences):
            self._compute_kmers_of_sequence(seq.seq._data, i)
            self.ids.append(seq.id)

        return self

    def __compute_kmers_from_file(self, sequences):
        path, ext = splitext(sequences.data)
        ext = ext.lstrip(".")
        fileList = []

        if not os.path.isdir(self.path):
            os.mkdir(self.path)

        cmd_split = '{} byname {} {}'.format(self.faSplit, sequences.data, self.path)

        os.system(cmd_split)

        for i, id in enumerate(sequences.ids):
            file = self.path + id + '.fa'
            fileList.append(file)

        with parallel_backend('dask'):
            Parallel(verbose = 100)(
            delayed(self._compute_kmers_of_sequence)(file, i)
            for i, file in enumerate(fileList))

        rmtree(self.path)
        return self

    def __compute_kmers_from_strings(self, sequences):
        for i, seq in enumerate(sequences):
            self._compute_kmers_of_sequence(seq, i)
            self.ids.append(i)

        return self

    def _compute_kmers(self, sequences):
        if isinstance(sequences, SeqCollection):
            if os.path.isfile(sequences.data):
                self.__compute_kmers_from_file(sequences)
            else:
                self.__compute_kmers_from_collection(sequences)
        else:
            self.__compute_kmers_from_strings(sequences)

    @abstractmethod
    def _compute_kmers_of_sequence(self, seq, i):
        """
        """

    def _convert_to_sparse_matrix(self):
        if self.sparse == "csr":
            self.data = csr_matrix(self.data, dtype=self.dtype)

        elif self.sparse == "csc":
            self.data = csc_matrix(self.data, dtype=self.dtype)


class SeenKmersCollection(KmersCollection):

    def __init__(self, sequences, Xy_file, length, k=5, sparse=None,
            dtype=np.uint64, alphabet="ACGT"):
        self.k = int(k)
        self.sparse = sparse
        self.dtype = dtype
        self.alphabet = alphabet.lower() + alphabet.upper()
        self.path = os.path.split(Xy_file)[0] + "/tmp/"
        self.Xy_file = tb.open_file(Xy_file, "w")
        self.length = length
        #
        self.ids = []
        self.v_size = 0
        self.dict_data = defaultdict(lambda: [0]*self.length)
        self.kmers_list = []
        self.data = "array"
        self.kmc_path = "{}/KMC/bin".format(os.path.dirname(os.path.realpath(__file__)))
        self.faSplit = "{}/faSplit".format(os.path.dirname(os.path.realpath(__file__)))
        self.client = Client(processes=False)
        #
        self._compute_kmers(sequences)
        self.__construct_data()
        self.Xy_file.close()

    def _compute_kmers_of_sequence(self, file, ind):
        print("Seen ind : ", ind)
        # Count k-mers with KMC
        cmd_count = "{}/kmc -k{} -fm -cs1000000000 -t48 -hp -sm -m1024 {} {}/{} {}".format(self.kmc_path, self.k, file, self.path, ind, self.path)
        run(cmd_count, shell = True, capture_output=True)
        # Transform k-mers db with KMC
        cmd_transform = "{}/kmc_tools transform {}/{} dump {}/{}.txt".format(self.kmc_path, self.path, ind, self.path, ind)
        run(cmd_transform, shell = True, capture_output=True)
        # Parse k-mers file to pandas
        profile = np.loadtxt('{}/{}.txt'.format(self.path, ind), dtype = object)
        # Save to Xyfile
        for row in profile:
            self.dict_data[row[0]][ind] = int(row[1])

        return self

    def __construct_data(self):
        # Get Kmers list
        self.kmers_list = list(self.dict_data)
        self.v_size = len(self.kmers_list)

        # Convert to numpy array and write directly to disk
        self.data = self.Xy_file.create_carray("/", "data", obj = np.array([ self.dict_data[x] for x in self.dict_data ],dtype=self.dtype).T)
        return self

class GivenKmersCollection(KmersCollection):

    def __init__(self, sequences, Xy_file, length, kmers_list, sparse=None,
            dtype=np.uint64, alphabet="ACGT"):
        self.sparse = sparse
        self.dtype = dtype
        self.alphabet = alphabet.lower() + alphabet.upper()
        self.path = os.path.split(Xy_file)[0] + "/tmp/"
        self.kmers_list = kmers_list
        self.Xy_file = tb.open_file(Xy_file, "w")
        self.length = length
        #
        self.k = len(self.kmers_list[0])
        self.__construct_kmer_indices()
        #
        self.ids = []
        self.v_size = len(self.kmers_list)
        self.dict_data = defaultdict(lambda: [0]*self.length)
        self.data = "array"
        self.kmc_path = "{}/KMC/bin".format(os.path.dirname(os.path.realpath(__file__)))
        self.faSplit = "{}/faSplit".format(os.path.dirname(os.path.realpath(__file__)))
        self.client = Client(processes=False)
        #
        self._compute_kmers(sequences)
        self.__construct_data()
        self.Xy_file.close()

    def _compute_kmers_of_sequence(self, file, ind):
        print("Given ind : ", ind)
        # Count k-mers with KMC
        cmd_count = "{}/kmc -k{} -fm -cs1000000000 -t48 -hp -sm -m1024 {} {}/{} {}".format(self.kmc_path, self.k, file, self.path, ind, self.path)
        run(cmd_count, shell = True, capture_output=True)
        # Transform k-mers db with KMC
        cmd_transform = "{}/kmc_tools transform {}/{} dump {}/{}.txt".format(self.kmc_path, self.path, ind, self.path, ind)
        run(cmd_transform, shell = True, capture_output=True)
        # Parse k-mers file to pandas
        profile = np.loadtxt('{}/{}.txt'.format(self.path, ind), dtype = object)

        for kmer in self.kmers_list:
            ind_kmer = self.kmers_list.index(kmer)
            for row in profile:
                if row[0] == kmer:
                    self.dict_data[row[0]][ind] = int(row[1])
                else:
                    self.dict_data[row[0]][ind] = 0

        return self

    def __construct_kmer_indices(self):
        self.kmers_indices = {kmer:i for i, kmer in enumerate(self.kmers_list)}

        return self

    def __construct_data(self):
        # Convert to numpy array and write directly to disk
        self.data = self.Xy_file.create_carray("/", "data", obj = np.array([ self.dict_data[x] for x in self.dict_data ],dtype=self.dtype).T)
        return self


# #####
# Data build functions
# ####################

def build_kmers(seq_data, k, Xy_file, length = 0, sparse=None, dtype=np.uint64):
    return SeenKmersCollection(seq_data, Xy_file, length, k=k, sparse=sparse, dtype=dtype)


def build_kmers_Xy_data(seq_data, k, Xy_file, length = 0, kmers_list = None, sparse=None, dtype=np.uint64):

    if kmers_list is not None:
        collection = GivenKmersCollection(seq_data, Xy_file, length, kmers_list, sparse)
    else:
        collection = build_kmers(seq_data, k, Xy_file, length, sparse)
    kmers_list = collection.kmers_list
    X_data = collection.data
    y_data = np.array(seq_data.labels)
    ids = seq_data.ids
    return X_data, y_data, kmers_list

def build_kmers_X_data(seq_data, X_file, kmers_list, length = 0, sparse=None, dtype=np.uint64):

    collection = GivenKmersCollection(seq_data, X_file, length, kmers_list, sparse)
    kmers_list = collection.kmers_list
    X_data = collection.data
    ids = seq_data.ids

    return X_data, kmers_list, ids
