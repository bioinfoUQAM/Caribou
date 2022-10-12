#!/usr/bin python3

import numpy as np
import pandas as pd

import os
import sys
import gzip
import tarfile
import argparse

from Bio import SeqIO
from data.build_data import build_load_save_data

__author__ = "Nicolas de Montigny"

__all__ = ['ReadsSimulation']

# Reduce number of cpus used to reduce nb of tmp files
# reduce number of reads generated

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

    def __init__(self, fasta, cls, genomes, sequencing, outdir, name):
        # Parameters
        if isinstance(fasta, tuple):
            self._fasta_in = fasta[0]
            self._fasta_host = fasta[1]
        else:
            self._fasta_in = fasta
            self._fasta_host = None
        self._cls_in = cls
        self._genomes = genomes
        self._nb_reads = len(genomes) * 10
        self._sequencing = sequencing
        self._path = outdir
        self._name = name
        self._prefix = os.path.join(outdir,'sim_{}'.format(self._name))
        # Files paths
        self._fasta_tmp = os.path.join(outdir, 'sim_{}_tmp.fasta'.format(self._name))
        self._R1_fastq = os.path.join(outdir, 'sim_{}_R1.fastq.gz'.format(self._name))
        self._R2_fastq = os.path.join(outdir, 'sim_{}_R2.fastq.gz'.format(self._name))
        self._fasta_out = os.path.join(outdir, 'sim_{}_data.fna.gz'.format(self._name))
        self._cls_out = os.path.join(outdir, 'sim_{}_class.csv'.format(self._name))
        # Dataset variables
        self.kmers_data = {}

    def simulation(self, k, kmers_list):
        self._make_tmp_fasta()
        cmd = "iss generate -g {} -n {} --abundance halfnormal --model {} --output {} --compress --cpus {}".format(self._fasta_tmp,self._nb_reads,self._sequencing,self._prefix,os.cpu_count())
        os.system(cmd)
        self._fastq2fasta()
        self._write_cls_file()
        self._kmers_dataset(k, kmers_list)
        return self.kmers_data


    def _make_tmp_fasta(self):
        for file in [self._fasta_in, self._fasta_host]:
            if file is not None:
                if os.path.splitext(file)[1] == '.gz':
                    self._add_tmp_fasta_gz(file)
                else:
                    self._add_tmp_fasta_fa(file)

    def _add_tmp_fasta_fa(self, file):
        with open(file, 'rt') as handle_in, open(self._fasta_tmp, 'at') as handle_out:
            for record in SeqIO.parse(handle_in, 'fasta'):
                if record.id in self._genomes:
                    SeqIO.write(record, handle_out, 'fasta')

    def _add_tmp_fasta_gz(self, file):
        with gzip.open(file, 'rt') as handle_in, open(self._fasta_tmp, 'at') as handle_out:
            for record in SeqIO.parse(handle_in, 'fasta'):
                if record.id in self._genomes:
                    SeqIO.write(record, handle_out, 'fasta')

    def _fastq2fasta(self):
        with gzip.open(self._R1_fastq, "rt") as handle_R1, gzip.open(self._R2_fastq, "rt") as handle_R2, gzip.open(self._fasta_out, "at") as handle_out:
            for record_R1, record_R2 in zip(SeqIO.parse(handle_R1, 'fastq'), SeqIO.parse(handle_R2, 'fastq')):
                record_R1.id = record_R1.id.replace('/','_')
                record_R2.id = record_R2.id.replace('/','_')
                SeqIO.write(record_R1, handle_out, 'fasta')
                SeqIO.write(record_R2, handle_out, 'fasta')

    def _write_cls_file(self):
        with gzip.open(self._fasta_out, 'rt') as handle:
            reads_ids = [record.id for record in SeqIO.parse(handle, 'fasta')]
        reads_crop = list(self._cls_in['id'])
        reads_df = pd.DataFrame({'reads_id' : reads_ids, 'id': np.empty(len(reads_ids), dtype=object)})
        for id in reads_crop:
            reads_df.loc[reads_df['reads_id'].str.contains(id),'id'] = id
        cls_out = reads_df.join(self._cls_in.set_index('id'), on = 'id')
        cls_out = cls_out.drop('id', axis = 1)
        cls_out = cls_out.rename(columns = {'reads_id':'id'})
        cls_out.to_csv(self._cls_out, index = False)

    def _kmers_dataset(self, k, kmers_list):
        self.kmers_data = build_load_save_data(None,
            (self._fasta_out,self._cls_out),
            self._path,
            None,
            'cv_simulation_{}'.format(self._name),
            k = k,
            kmers_list = kmers_list
        )
