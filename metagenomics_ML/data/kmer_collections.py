from .seq_collections import SeqCollection

from abc import ABC, abstractmethod
import re
from collections import defaultdict
from itertools import product
from Bio import SeqIO

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.feature_selection import VarianceThreshold


__all__ = [ 'FullKmersCollection', 'SeenKmersCollection',
        'GivenKmersCollection' , 'VarKmersCollection',
        'build_kmers', 'build_kmers_Xy_data']

# From mlr_kgenomvir
__author__ = "Amine Remita" # Adapted by Nicolas de Montigny

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
        with gzip.open(sequences, "rt") as handle:
            records = SeqIO(handle, "fasta")
            i = 0
            while True:
                try:
                    record = next(records)
                    self.ids.append(record.id)
                    self._compute_kmers_of_sequence(record.seq._data, i)
                    i += 1
                except StopIteration:
                    False

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

    def __init__(self, sequences, k=5, sparse=None,
            dtype=np.uint64, alphabet="ACGT"):
        self.k = k
        self.sparse = sparse
        self.dtype = dtype
        self.alphabet = alphabet
        #
        self.ids = []
        self.v_size = np.power(len(self.alphabet), int(self.k))
        self.data = np.zeros((len(sequences), self.v_size), dtype=self.dtype)
        self.kmers_list = ["".join(t) for t in product(alphabet, repeat=k)]
        #
        self._compute_kmers(sequences)
        self._convert_to_sparse_matrix()

    def _compute_kmers_of_sequence(self, sequence, ind):
        search = re.compile("^["+self.alphabet+"]+$").search

        for i in range(len(sequence) - int(self.k) + 1):
            kmer = sequence[i:i + int(self.k)]

            if self.alphabet and bool(search(kmer)) or not self.alphabet:
                ind_kmer = get_index_from_kmer(kmer, int(self.k))
                self.data[ind][ind_kmer] += 1

        return self


class SeenKmersCollection(KmersCollection):

    def __init__(self, sequences, k=5, sparse=None,
            dtype=np.uint64, alphabet="ACGT"):
        self.k = int(k)
        self.sparse = sparse
        self.dtype = dtype
        self.alphabet = alphabet
        #
        self.ids = []
        self.v_size = 0
        self.data = []
        self.dict_data = defaultdict(lambda: [0]*len(sequences))
        self.kmers_list = []
        #
        self._compute_kmers(sequences)
        self.__construct_data()
        self._convert_to_sparse_matrix()

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

        # Convert to numpy
        self.data = np.array([ self.dict_data[x] for x in self.dict_data ],
                dtype=self.dtype).T

        return self


class VarKmersCollection(SeenKmersCollection):

    def __init__(self, sequences, low_var_threshold=0.0, k=5, sparse=None,
            dtype=np.uint64, alphabet="ACGT"):
        super().__init__(sequences, k=k, sparse=sparse,
                dtype=dtype, alphabet=alphabet)

        # Kmer selection based on variance
        selection = VarianceThreshold(threshold=low_var_threshold)
        self.data = selection.fit_transform(self.data)

        # update kmer list
        self.v_size = self.data.shape[1]
        _support = selection.get_support()
        self.kmers_list = [ kmer for i, kmer in enumerate(self.kmers_list) if _support[i] ]


class GivenKmersCollection(KmersCollection):

    def __init__(self, sequences, kmers_list, sparse=None,
            dtype=np.uint64, alphabet="ACGT"):
        self.sparse = sparse
        self.dtype = dtype
        self.alphabet = alphabet
        self.kmers_list = kmers_list
        #
        self.k = len(self.kmers_list[0])
        self.__construct_kmer_indices()
        #
        self.ids = []
        self.v_size = len(self.kmers_list)
        self.data = np.zeros((len(sequences), self.v_size), dtype=self.dtype)
        #
        self._compute_kmers(sequences)
        self._convert_to_sparse_matrix()

    def _compute_kmers_of_sequence(self, sequence, ind):
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]

            if kmer in self.kmers_indices:
                ind_kmer = self.kmers_indices[kmer]
                self.data[ind][ind_kmer] += 1

        return self

    def __construct_kmer_indices(self):
        self.kmers_indices = {kmer:i for i, kmer in enumerate(self.kmers_list)}

        return self


# #####
# Data build functions
# ####################

def build_kmers(seq_data, k, full_kmers=False, low_var_threshold=None,
        sparse=None, dtype=np.uint64):

    if full_kmers:
        return FullKmersCollection(
                seq_data, k=k, sparse=sparse, dtype=dtype)

    elif low_var_threshold:
        return VarKmersCollection(
                seq_data, low_var_threshold=low_var_threshold,
                k=k, sparse=sparse, dtype=dtype)
    else:
        return SeenKmersCollection(
                seq_data, k=k, sparse=sparse, dtype=dtype)


def build_kmers_Xy_data(seq_data, k, full_kmers=False, low_var_threshold=None,
        sparse=None, dtype=np.uint64):

    collection = build_kmers(seq_data, k, full_kmers, low_var_threshold, sparse, dtype)
    kmers_list = collection.kmers_list
    X_data = collection.data
    y_data = np.asarray(seq_data.labels)

    return X_data, y_data, kmers_list

def build_kmers_X_data(seq_file, k, full_kmers=False, low_var_threshold=None,
        sparse=None, dtype=np.uint64):

    ids = []
    sequences = []

    with open(seq_file) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            ids.append(record.id)
            sequences.append(str(record.seq))

    collection = build_kmers(sequences, k, full_kmers, low_var_threshold, sparse, dtype)
    kmers_list = collection.kmers_list
    X_data = collection.data

    return X_data, kmers_list, ids
