#!/usr/bin python3

import numpy as np
import pandas as pd

import os
import sys
import gzip
import tarfile
import argparse

from Bio import SeqIO, Entrez
from data.build_data import build_load_save_data

__author__ = "Nicolas de Montigny"

__all__ = ['ReadsSimulation']

class readsSimulation():
    """
    Class used to make reads simulation from whole genomes files

    ----------
    Parameters
    ----------

    fasta:
        string : Path to a fasta file containing the genomes to simulate reads from
    cls:
        pandas.dataframe.DataFrame : DataFrame containing the classe for each IDs of the original dataset
    genomes:
        list : A list of IDs to use for simulating reads, must be present in fasta and cls files
    sequencing:
        string : Type of Illumina sequencing to be simulated among : MiSeq, HiSeq and NovaSeq
    outdir:
        string : Path to a folder where the simulation data should be saved

    ----------
    Attributes
    ----------
        kmers_data :
            dictionnary : K-mers data as is constructed by the build_load_save_data method

    ----------
    Methods
    ----------
        simulation : initiate simulation and k-mers extraction for the simulated reads
            k : integer
                Length of the k-mers to extract, must be concordant with the database used for classification
            kmers_list : list of strings
                List of the k-mers to extract, must be concordant with the database used for classification

    """

    def __init__(self, fasta, cls, genomes, sequencing, outdir):
        # Parameters
        self._fasta_in = fasta
        self._cls_in = cls
        self._genomes = genomes
        self._nb_reads = len(genomes) * 20
        self._sequencing = sequencing
        self._path = outdir
        # Files paths
        self._fasta_tmp = os.path.join(self._path, 'sim_tmp.fasta')
        self._abund_file = os.path.join(self._path, 'sim_abundance.txt')
        self._R1_fastq = os.path.join(self._path, 'sim_R1.fastq.gz')
        self._R2_fastq = os.path.join(self._path, 'sim_R2.fastq.gz')
        self._fasta_out = os.path.join(self._path, 'sim_data.fna.gz')
        self._cls_out = os.path.join(self._path, 'sim_class.csv')
        # Dataset variables
        self.kmers_data = {}

    def simulation(self, k, kmers_list):
        self._make_tmp_fasta()
        cmd = "iss generate -g {} -u {} -n {} --abundance halfnormal --model {} --output {} --compress --cpus {}".format(self._fasta_tmp,len(self._genomes),self._nb_reads,self._sequencing,self._path,os.cpu_count())
        os.system(cmd)
        self._fastq2fasta()
        self._write_cls_file()
        self._kmers_dataset(k, kmers_list)

    def _make_tmp_fasta(self):
        with gzip.open(self._fasta_in, 'r') as handle_in, open(self._fasta_tmp, 'w') as handle_out:
            for record in SeqIO.parse(handle_in, 'fasta'):
                if record.id in self._genomes:
                    SeqIO.write(record, handle_out, 'fasta')


    def _fastq2fasta(self):
        with gzip.open(self._R1_fastq, "rt") as handle_R1, gzip.open(self._R2_fastq, "rt") as handle_R2, gzip.open(self._fasta_out, "at") as handle_out:
            for record_R1, record_R2 in zip(SeqIO.parse(handle_R1, 'fastq'), SeqIO.parse(handle_R2, 'fastq')):
                SeqIO.write(record_R1, handle_out, 'fasta')
                SeqIO.write(record_R2, handle_out, 'fasta')

# TODO: USED CLASSES FROM KMERS_DATA TO MASK IDS -> CREATE CSV FILE
    def _write_cls_file(self):
        reads_ids = pd.read_table(self._abund_file, header = None, names = ['id', 'abundance'])
        cls_out = pd.merge(self._cls_in, reads_ids, on = 'id', how = 'outer')
        cls_out.to_csv(self._cls_out, index = False)

    def _kmers_dataset(self, k, kmers_list):
        self.kmers_data = build_load_save_data(None,
            (self._fasta_out,self._cls_out),
            self._path,
            'cv_simulation',
            None,
            k = k,
            kmers_list = kmers_list
        )
