from .seq_collections import SeqCollection

import re
import os
import gzip

from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from Bio import SeqIO
from os.path import splitext

import numpy as np
import tables as tb
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.feature_selection import VarianceThreshold


__all__ = [ 'FullKmersCollection', 'SeenKmersCollection',
        'GivenKmersCollection' , 'VarKmersCollection',
        'build_kmers', 'build_kmers_Xy_data']

# From mlr_kgenomvir
__author__ = "Nicolas de Montigny"

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

#TESTER SI FONCTIONNE BIEN ET NE RESTE PAS EN LOOP INFINIE
    def __compute_kmers_from_file(self, sequences):
        path, ext = splitext(sequences)
        ext = ext.lstrip(".")

        if ext in ["fa","fna"]:
            ext = "fasta"
            with open(sequences, "rt") as handle:
                records = SeqIO.parse(handle, "fasta")
                i = 0
                error = False
                while not error:
                    try:
                        record = next(records)
                        self.ids.append(record.id)
                        self._compute_kmers_of_sequence(record.seq._data, i)
                        i += 1
                    except StopIteration as e:
                        error = True

        elif ext == "gz":
            path, ext = splitext(path)
            ext = ext.lstrip(".")
            if ext in ["fa","fna"]:
                ext = "fasta"

            with gzip.open(sequences, "rt") as handle:
                records = SeqIO.parse(handle, "fasta")
                i = 0
                error = False
                while not error:
                    try:
                        record = next(records)
                        self.ids.append(record.id)
                        self._compute_kmers_of_sequence(record.seq._data, i)
                        i += 1
                    except StopIteration as e:
                        error = True

        return self

    def __compute_kmers_from_strings(self, sequences):
        for i, seq in enumerate(sequences):
            self._compute_kmers_of_sequence(seq, i)
            self.ids.append(i)

        return self

    def _compute_kmers(self, sequences):
        if isinstance(sequences, SeqCollection):
            if os.path.isfile(sequences.data):
                self.__compute_kmers_from_file(sequences.data)
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


class FullKmersCollection(KmersCollection):

    def __init__(self, sequences, Xy_file, length, k=5, sparse=None,
            dtype=np.uint64, alphabet="ACGT"):
        self.k = k
        self.sparse = sparse
        self.dtype = dtype
        self.alphabet = alphabet
        self.Xy_file = tb.open_file(Xy_file, "w")
        self.length = length
        #
        self.ids = []
        self.v_size = np.power(len(self.alphabet), int(self.k))
        self.data = self.Xy_file.create_carray("/", "data", obj = np.zeros((self.length, self.v_size)))
        self.kmers_list = ["".join(t) for t in product(alphabet, repeat=k)]
        #
        self._compute_kmers(sequences)
        self.Xy_file.close()

    def _compute_kmers_of_sequence(self, sequence, ind):
        search = re.compile("^["+self.alphabet+"]+$").search

        for i in range(len(sequence) - int(self.k) + 1):
            kmer = sequence[i:i + int(self.k)]

            if self.alphabet and bool(search(kmer)) or not self.alphabet:
                ind_kmer = get_index_from_kmer(kmer, int(self.k))
                self.data[ind,ind_kmer] += 1

        return self


class SeenKmersCollection(KmersCollection):

    def __init__(self, sequences, Xy_file, length, k=5, sparse=None,
            dtype=np.uint64, alphabet="ACGT"):
        self.k = int(k)
        self.sparse = sparse
        self.dtype = dtype
        self.alphabet = alphabet
        self.Xy_file = tb.open_file(Xy_file, "w")
        self.length = length
        #
        self.ids = []
        self.v_size = 0
        self.dict_data = defaultdict(lambda: [0]*self.length)
        self.kmers_list = []
        self.data = "array"
        #
        self._compute_kmers(sequences)
        self.__construct_data()
        self.Xy_file.close()

    def _compute_kmers_of_sequence(self, sequence, ind):
        search = re.compile("^["+self.alphabet+"]+$").search

        for i in range(len(sequence) - int(self.k) + 1):
            kmer = sequence[i:i + int(self.k)]

            if self.alphabet and bool(search(kmer)) or not self.alphabet:
                self.dict_data[kmer][ind] += 1

        return self

    def __construct_data(self):
        # Get Kmers list
        self.kmers_list = list(self.dict_data)
        self.v_size = len(self.kmers_list)

        # Convert to numpy array and write directly to disk
        self.data = self.Xy_file.create_carray("/", "data", obj = np.array([ self.dict_data[x] for x in self.dict_data ],dtype=self.dtype).T)

        return self


class VarKmersCollection(SeenKmersCollection):

    def __init__(self, sequences, Xy_file, length = 0, low_var_threshold=0.01, k=5, sparse=None,
            dtype=np.uint64, alphabet="ACGT"):
        super().__init__(sequences, Xy_file = Xy_file, length = length, k=k, sparse=sparse,
                dtype=dtype, alphabet=alphabet)

        # Kmer selection based on variance
        selection = VarianceThreshold(threshold=low_var_threshold)
        with tb.open_file(Xy_file, "r") as handle:
            self.data = selection.fit_transform(handle.get_node("/", "data").read())

        # update kmer list
        self.v_size = self.data.shape[1]
        _support = selection.get_support()
        self.kmers_list = [ kmer for i, kmer in enumerate(self.kmers_list) if _support[i] ]
        with tb.open_file(Xy_file, "w") as handle:
            self.data = handle.create_carray("/", "data", obj = np.array(self.data,dtype=self.dtype))


class GivenKmersCollection(KmersCollection):

    def __init__(self, sequences, Xy_file, length, kmers_list, sparse=None,
            dtype=np.uint64, alphabet="ACGT"):
        self.sparse = sparse
        self.dtype = dtype
        self.alphabet = alphabet
        self.kmers_list = kmers_list
        self.Xy_file = tb.open_file(Xy_file, "w")
        self.length = length
        #
        self.k = len(self.kmers_list[0])
        self.__construct_kmer_indices()
        #
        self.ids = []
        self.v_size = len(self.kmers_list)
        self.data = self.Xy_file.create_carray("/", "data", obj = np.zeros(self.length, self.v_size), dtype=self.dtype)
        #
        self._compute_kmers(sequences)
        self.Xy_file.close()

    def _compute_kmers_of_sequence(self, sequence, ind):
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]

            if kmer in self.kmers_indices:
                ind_kmer = self.kmers_indices[kmer]
                self.data[ind,ind_kmer] += 1

        return self

    def __construct_kmer_indices(self):
        self.kmers_indices = {kmer:i for i, kmer in enumerate(self.kmers_list)}

        return self


# #####
# Data build functions
# ####################

def build_kmers(seq_data, k, Xy_file, length = 0, full_kmers=False, low_var_threshold=None,
        sparse=None, dtype=np.uint64):

    if full_kmers:
        return FullKmersCollection(
                seq_data, Xy_file, length, k=k, sparse=sparse, dtype=dtype)

    elif low_var_threshold:
        return VarKmersCollection(
                seq_data, Xy_file, length, low_var_threshold=low_var_threshold,
                k=k, sparse=sparse, dtype=dtype)
    else:
        return SeenKmersCollection(
                seq_data, Xy_file, length, k=k, sparse=sparse, dtype=dtype)


def build_kmers_Xy_data(seq_data, k, Xy_file, length = 0, full_kmers=False, low_var_threshold=None,
        sparse=None, dtype=np.uint64):

    collection = build_kmers(seq_data, k, Xy_file, length, full_kmers, low_var_threshold, sparse, dtype)
    kmers_list = collection.kmers_list
    X_data = collection.data
    y_data = np.asarray(seq_data.labels)

    return X_data, y_data, kmers_list

def build_kmers_X_data(seq_data, k, X_file, length = 0, full_kmers=False, low_var_threshold=None,
        sparse=None, dtype=np.uint64):

    collection = build_kmers(seq_data, k, X_file, length, full_kmers, low_var_threshold, sparse, dtype)
    kmers_list = collection.kmers_list
    X_data = collection.data
    ids = collection.ids

    return X_data, kmers_list, ids
