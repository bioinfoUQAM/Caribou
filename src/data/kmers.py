import os
import ray
import gzip
import warnings

import numpy as np
import pandas as pd

from Bio import SeqIO
from glob import glob
from shutil import rmtree
from os.path import splitext

# Kmers extraction
from data.extraction.seen_kmers_vectorizer import SeenKmersVectorizer
from data.extraction.given_kmers_vectorizer import GivenKmersVectorizer

__author__ = ['Amine Remita', 'Nicolas de Montigny']

__all__ = ['KmersCollection']

"""
Module inspired from module kmer_collections.py of
mlr_kgenomvir package [Remita et al. 2022]

Load sequences to pandas dataframe by batch then saved to parquet files.
Read parquet files into a unified ray dataset, before tokenizing kmers from sequence into count matrix and concatenating into a tensor.
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

    csv : string
        A csv file containing all classes in the database associated to each ID

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
        fasta_file,
        Xy_file,
        k,
        cls_file = None,
        kmers_list = None,
    ):
        ## Public attributes
        # Parameters
        self.k = k
        self.Xy_file = Xy_file
        self.fasta = fasta_file
        self.csv = cls_file
        # Initialize variables
        self.df = None
        self.ids = []
        self.taxas = []
        self.method = None
        self.kmers_list = None
        self._nb_kmers = 0
        self._labels = None
        self._files_list = []
        self.memory_parsing = False
        
        # Infer method from presence of already extracted kmers or not
        if isinstance(kmers_list, list):
            self.method = 'given'
            self.kmers_list = kmers_list
            self._nb_kmers = len(self.kmers_list)
        else:
            self.method = 'seen'
        
        # Global tmp dir path
        self._tmp_dir = os.path.join(os.path.split(Xy_file)[0],"tmp","")
        # Make global tmp dir if it doesn't exist
        if not os.path.isdir(self._tmp_dir):
            os.mkdir(self._tmp_dir)

        # Read classes files if present
        if self.csv is not None:
            self._read_cls_file()

    def _read_cls_file(self):
        self._labels = pd.read_csv(self.csv)
        # Get taxas from csv file
        if len(self._labels.columns) > 0:
            self.taxas = list(self._labels.columns)
            self.taxas.remove('id')
        else:
            raise ValueError(f'No information found in the classes csv file : {self.csv}')

    # Execute k-mers extraction
    def compute_kmers(self):
        print('compute_kmers')
        self._verif_mem_vs_disk()
        self._parse_fasta()
        self._make_ray_ds()
        self._kmers_tokenization()
        self._write_dataset()

    def _verif_mem_vs_disk(self):
        mem = ray.cluster_resources()['memory']
        fasta_size = os.path.getsize(os.path.expanduser(self.fasta))
        if mem > fasta_size:
            self.memory_parsing = True

    def _parse_fasta(self):
        print('_parse_fasta')
        if os.path.isfile(self.fasta):
            if self.memory_parsing:
                self._single_fasta_ds_mem()
            else:
                self._single_fasta_ds_disk()
        elif os.path.isdir(self.fasta):
            if self.memory_parsing:
                self._multi_fasta_ds_mem()
            else:
                self.fasta = glob(os.path.join(self.fasta, '*.fa'))
                self._multi_fasta_ds_disk()
        else:
            raise ValueError('Fasta must be an interleaved fasta file or a directory containing fasta files.')
    
    def _single_fasta_ds_mem(self):
        print('_single_fasta_ds_mem')
        data = {
            'id':[],
            'sequence':[]
        }
        path, ext = splitext(self.fasta)
        ext = ext.lstrip(".")
        if ext in ["fa","fna","fasta"]:
            with open(self.fasta, 'rt') as handle:
                for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
                    data['id'].append(record.id)
                    data['sequence'].append(str(record.seq).upper())
        elif ext == "gz":
            with gzip.open(self.fasta, 'rt') as handle:
                for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
                    data['id'].append(record.id)
                    data['sequence'].append(str(record.seq).upper())
        else:
            raise ValueError(f'Unknown file extension : {ext}')
        
        self.ids = data['id']
        self.df = pd.DataFrame(data)
        if self._labels is not None:
            self.df = pd.merge(self.df, self._labels, on = 'id', how = 'left')

    def _single_fasta_ds_disk(self):
        print('_single_fasta_ds_disk')
        data = {
            'id':[],
            'sequence':[]
        }
        path, ext = splitext(self.fasta)
        ext = ext.lstrip(".")
        if ext in ["fa","fna","fasta"]:
            with open(self.fasta, 'rt') as handle:
                for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
                    data['id'].append(record.id)
                    data['sequence'].append(str(record.seq).upper())
                    if i % 100 == 0 :
                        df = pd.DataFrame(data)
                        if self._labels is not None:
                            cls = self._labels[self._labels['id'].isin(data['id'])]
                            df = pd.merge(df, cls, on = 'id', how = 'left')
                        df.to_parquet(os.path.join(self._tmp_dir, f'batch_{int(i/100)}.parquet'))
                        self.ids.extend(data['id'])
                        data = {
                            'id':[],
                            'sequence':[]
                        }
                if len(data['id']) != 0:
                    df = pd.DataFrame(data)
                    if self._labels is not None:
                        cls = self._labels[self._labels['id'].isin(data['id'])]
                        df = pd.merge(df, cls, on = 'id', how = 'left')
                    df.to_parquet(os.path.join(self._tmp_dir, f'batch_end.parquet'))
                    self.ids.extend(data['id'])
        elif ext == "gz":
            with gzip.open(self.fasta, 'rt') as handle:
                for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
                    data['id'].append(record.id)
                    data['sequence'].append(str(record.seq).upper())
                    if i % 100 == 0 :
                        df = pd.DataFrame(data)
                        if self._labels is not None:
                            cls = self._labels[self._labels['id'].isin(data['id'])]
                            df = pd.merge(df, cls, on = 'id', how = 'left')
                        df.to_parquet(os.path.join(self._tmp_dir, f'batch_{int(i/100)}.parquet'))
                        self.ids.extend(data['id'])
                        data = {
                            'id':[],
                            'sequence':[]
                        }
                if len(data['id']) != 0:
                    df = pd.DataFrame(data)
                    if self._labels is not None:
                        cls = self._labels[self._labels['id'].isin(data['id'])]
                        df = pd.merge(df, cls, on = 'id', how = 'left')
                    df.to_parquet(os.path.join(self._tmp_dir, f'batch_end.parquet'))
                    self.ids.extend(data['id'])
        else:
            raise ValueError(f'Unknown file extension : {ext}')

    def _multi_fasta_ds_mem(self):
        print('_multi_fasta_ds_mem')
        data = {
            'id':[],
            'sequence':[]
        }
        for i, file in enumerate(self.fasta):
            path, ext = splitext(file)
            ext = ext.lstrip(".")
            if ext in ["fa","fna","fasta"]:
                with open(file, 'rt') as handle:
                    for record in SeqIO.parse(handle, 'fasta'):
                        data['id'].append(record.id)
                        data['sequence'].append(str(record.seq).upper())
            elif ext == "gz":
                with gzip.open(file, 'rt') as handle:
                    for record in SeqIO.parse(handle, 'fasta'):
                        data['id'].append(record.id)
                        data['sequence'].append(str(record.seq).upper())
            else:
                raise ValueError(f'Unknown file extension : {ext}')
        
        self.ids = data['id']
        self.df = pd.DataFrame(data)
        if self._labels is not None:
            self.df = pd.merge(self.df, self._labels, on = 'id', how = 'left')
        
    def _multi_fasta_ds_disk(self):
        print('_multi_fasta_ds_disk')
        data = {
            'id':[],
            'sequence':[]
        }
        for i, file in enumerate(self.fasta):
            path, ext = splitext(file)
            ext = ext.lstrip(".")
            if ext in ["fa","fna","fasta"]:
                with open(file, 'rt') as handle:
                    for record in SeqIO.parse(handle, 'fasta'):
                        data['id'].append(record.id)
                        data['sequence'].append(str(record.seq).upper())
            elif ext == "gz":
                with gzip.open(file, 'rt') as handle:
                    for record in SeqIO.parse(handle, 'fasta'):
                        data['id'].append(record.id)
                        data['sequence'].append(str(record.seq).upper())
            else:
                raise ValueError(f'Unknown file extension : {ext}')
            if i % 100 == 0 :
                df = pd.DataFrame(data)
                if self._labels is not None:
                    cls = self._labels[self._labels['id'].isin(data['id'])]
                    df = pd.merge(df, cls, on = 'id', how = 'left')
                df.to_parquet(os.path.join(self._tmp_dir, f'batch_{int(i/100)}.parquet'))
                self.ids.extend(data['id'])
                data = {
                    'id':[],
                    'sequence':[]
                }
        if len(data['id']) != 0:
            df = pd.DataFrame(data)
            if self._labels is not None:
                cls = self._labels[self._labels['id'].isin(data['id'])]
                df = pd.merge(df, cls, on = 'id', how = 'left')
            df.to_parquet(os.path.join(self._tmp_dir, f'batch_end.parquet'))
            self.ids.extend(data['id'])

    def _make_ray_ds(self):
        print('_make_ray_ds')
        if self.memory_parsing:
            self.df = ray.data.from_pandas(self.df)
            if self.df.count() > 10:
                self.df = self.df.repartition(int(self.df.count()/10))
        else:
            self._files_list = glob(os.path.join(self._tmp_dir, '*.parquet'))
            self.df = ray.data.read_parquet_bulk(self._files_list, parallelism = len(files_lst))

    def _kmers_tokenization(self):
        print('_kmers_tokenization')
        if self.method == 'seen':
            tokenizer = SeenKmersVectorizer(
                k = self.k,
                column = 'sequence'
            )
        elif self.method == 'given':
            tokenizer = GivenKmersVectorizer(
                k = self.k,
                column = 'sequence',
                tokens = self.kmers_list
            )
        tokenizer.fit(self.df)
        self.df = tokenizer.transform(self.df)
        if self.method == 'seen':
            self.kmers_list = tokenizer.stats_['tokens(sequence)']
 
    def _write_dataset(self):
        self.df.write_parquet(self.Xy_file)
        rmtree(self._tmp_dir)
