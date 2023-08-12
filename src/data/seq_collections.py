
import copy
import gzip
import random

import pandas as pd

from Bio import SeqIO
from glob import glob
from Bio.SeqRecord import SeqRecord
from collections import UserList, defaultdict
from os.path import splitext, isfile, isdir, join
from joblib import Parallel, delayed, parallel_backend

# From mlr_kgenomvir
__author__ = ['Amine Remita', 'Nicolas de Montigny']

__all__ = ['SeqCollection']

"""
Module adapted from module seq_collections.py of
mlr_kgenomvir package [Remita et al. 2022]

Keeps SeqRecords in file instead parsing them to memory
and adapted / added functions to do so.
"""

class SeqCollection(UserList):

    """
    Attributes
    ----------

    data : string
        Path to a file containing sequences in fasta format

    labels : list of strings
        Collection of labels of the sequences
        The order of label needs to be the same as
        the sequences in data

    label_map : dict
        mapping of sequences and their labels (classes)


    """

# Fasta file path
# Taxas
# Labels

    def __init__(self, arg):

        self.data = []
        self.labels = []
        self.label_map = {}
        self.label_ind = defaultdict(list)
        self.ids = []
        self.id_map = {}
        self.id_ind = defaultdict(list)
        self.taxas = []

        # If arguments are two files
        # Fasta file and annotation file
        if isinstance(arg, tuple):
            self.labels, self.taxas = self.read_class_file(arg[1])
            self.data = self.read_bio_file(arg[0])
            self._verify_remove_duplicates()

        elif isfile(arg):
            self.data = self.read_bio_file(arg)

        # If argument is a list of labeled seq records
        elif isinstance(arg, list):
            self.data = copy.deepcopy(arg)
            self.__get_labels()
            self.__set_ids()

        # If argument is SeqCollection object
        elif isinstance(arg, self.__class__):
            self.data = copy.deepcopy(arg.data)
            self.__get_labels()
            self.__set_ids()

        # why?
        else:
            self.data = list(copy.deepcopy(arg))
            self.__get_labels()
            self.__set_ids()

    def __set_labels(self):
        for ind, seqRecord in enumerate(self.data):
            if seqRecord.id in self.label_map:
                seqRecord.label = self.label_map[seqRecord.id]
                self.labels.append(self.label_map[seqRecord.id])
                self.label_ind[seqRecord.label].append(ind)

            else:
                print("No label for {}\n".format(seqRecord.id))
                self.labels.append("UNKNOWN")
                self.label_ind["UNKNOWN"].append(ind)

    def __append_label(self, id, ind):
        if id in self.label_map:
            self.labels[ind,:]
            #self.labels.append(self.label_map[id])
            self.label_ind[self.label_map[id]].append(ind)
        elif not self.label_map:
            self.labels.append("UNKNOWN")
            self.label_ind["UNKNOWN"].append(ind)
        else:
            print("No label for {}\n".format(id))
            self.labels.append("UNKNOWN")
            self.label_ind["UNKNOWN"].append(ind)

    def __get_labels(self):

        self.label_map = dict((seqRec.id, seqRec.label)
                        for seqRec in self.data)

        self.labels = list(seqRec.label for seqRec in self.data)

        for ind, seqRecord in enumerate(self.data):
            self.label_ind[seqRecord.label].append(ind)

    def __set_ids(self):
        for seqRecord in self.data:
            self.ids.append(seqRecord.id)

    def __append_id(self, id):
        self.ids.append(id)

    def __getitem__(self, ind):
        # TODO
        # Give more details about this exception
        if not isinstance(ind, (int, list, slice)):
            raise TypeError("The argument must be int, list or slice")

        # shallow copy
        #if the argument is an integer
        if isinstance(ind, int):
            return self.data[ind]

        # With instantiation, data will be deep copied
        # If the argument is a list of indices
        elif isinstance(ind, list):

            tmp = [self.data[i] for i in ind if i>= 0 and i<len(self.data)]
            return self.__class__(tmp)

        return self.__class__(self.data[ind])

    def read_bio_file(self, file):
        if isfile(file):
            file = self._read_bio_file(file)
        elif isdir(file):
            file = self._read_bio_folder(file)
        
        return file

    def _read_bio_folder(self, dir):
        files_lst = []
        for ext in ['.fa', '.fna', '.fasta','.gz']:
            files_lst.extend(glob(join(dir, f'*{ext}')))
        
        with parallel_backend('threading'):
            files_lst = Parallel(n_jobs=-1, prefer = 'threads', verbose = 1)(
                delayed(self._read_bio_file)
                (file) for file in files_lst)
        return files_lst

    def _read_bio_file(self, file):
        path, ext = splitext(file)
        ext = ext.lstrip(".")

        if ext in ["fa","fna"]:
            ext = "fasta"
            with open(file, "r") as handle_in:
                records = SeqIO.parse(handle_in, ext)
                error = False
                while not error:
                    try:
                        record = next(records)
                        self.__append_id(record.id)
                    except StopIteration as e:
                        error = True
        elif ext == "gz":
            path, ext = splitext(path)
            ext = ext.lstrip(".")
            if ext in ["fa","fna"]:
                ext = "fasta"
            with gzip.open(file, "rt") as handle_in:
                records = SeqIO.parse(handle_in, ext)
                error = False
                while not error:
                    try:
                        record = next(records)
                        self.__append_id(record.id)
                    except StopIteration as e:
                        error = True

        return file

    def _verify_remove_duplicates(self):
        self.labels # classes
        self.taxas # columns
        self.ids # ids list
        df = pd.DataFrame(
            self.labels,
            columns = self.taxas,
            index = self.ids
        )
        df = df.reset_index(names = 'id')
        df = df.drop_duplicates(
            subset = 'id',
            keep = 'first',
            ignore_index = True
        )
        self.ids = df.id.values
        df = df.drop('id', axis = 1)
        self.labels = df.to_numpy()

    @classmethod
    def read_class_file(cls, file):
        csv = pd.read_csv(file, header = 0)
        return csv.iloc[:,1:].to_numpy(), list(csv.iloc[:,1:].columns)

    @classmethod
    def write_fasta(cls, data, out_fasta):
        SeqIO.write(data, out_fasta, "fasta")

    @classmethod
    def write_classes(cls, classes, file_class):
        with open(file_class, "w") as fh:
            for entry in classes:
                fh.write(entry+","+classes[entry]+"\n")

    def extract_fragments(self, size, stride=1):

        if stride < 1:
            print("extract_fragments() stride parameter should be sup to 1")
            stride = 1

        new_data = []

        for ind, seqRec in enumerate(self.data):
            sequence = seqRec.seq

            i = 0
            j = 0
            while i < (len(sequence) - size + 1):
                fragment = sequence[i:i + size]

                frgRec = SeqRecord(fragment, id=seqRec.id + "_" + str(j))
                frgRec.rankParent = ind
                frgRec.idParent = seqRec.id
                frgRec.label = seqRec.label
                frgRec.description = seqRec.description
                frgRec.name = "{}.fragment_at_{}".format(seqRec.name, str(i))
                frgRec.position = i

                new_data.append(frgRec)
                i += stride
                j += 1

        return self.__class__(new_data)

    def get_parents_rank_list(self):
        parents = defaultdict(list)

        for ind, seqRec in enumerate(self.data):
            if hasattr(seqRec, "rankParent"):
                parents[seqRec.rankParent].append(ind)

        return parents

    def sample(self, size, seed=None):
        random.seed(seed)

        if size > len(self.data):
            return self

        else:
            return self.__class__(random.sample(self, size))

    def stratified_sample(self, sup_limit=25, inf_limit=5, seed=None):
        random.seed(seed)

        new_data_ind = []

        for label in self.label_ind:
            nb_seqs = len(self.label_ind[label])
            the_limit = sup_limit

            if nb_seqs <= the_limit:
                the_limit = nb_seqs

            if nb_seqs >= inf_limit:
                new_data_ind.extend(random.sample(self.label_ind[label], the_limit))

        return self[new_data_ind]

    def get_count_labels(self):
        count = {label:len(self.label_ind[label])
                for label in self.label_ind}

        return count

    def write(self, fasta_file, class_file):
       self.write_fasta(self.data, fasta_file)
       self.write_classes(self.label_map, class_file)
