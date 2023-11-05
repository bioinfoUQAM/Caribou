#!/usr/bin python3

import numpy as np
import pandas as pd

import os
import gzip

from Bio import SeqIO
from glob import glob
from pathlib import Path
from warnings import warn
from data.build_data import build_load_save_data
from joblib import Parallel, delayed, parallel_backend

__author__ = "Nicolas de Montigny"

__all__ = ['ReadsSimulation','split_sim_dataset','sim_dataset']

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
            Optional parameters, they must be specified together or not at all
            k : integer
                Length of the k-mers to extract, must be concordant with the database used for classification
            kmers_list : list of strings
                List of the k-mers to extract, must be concordant with the database used for classification

    """

    def __init__(
        self,
        fasta,
        cls,
        genomes,
        sequencing,
        outdir,
        name
    ):
        # Parameters
        if isinstance(fasta, tuple):
            self._fasta_in = fasta[0]
            self._fasta_host = fasta[1]
        else:
            self._fasta_in = fasta
            self._fasta_host = None
        self._cls_in = cls
        self._genomes = genomes
        self._nb_reads = len(genomes) * 5
        self._sequencing = sequencing
        self._path = outdir
        self._tmp_path = os.path.join(outdir,'tmp')
        self._name = name
        self._prefix = os.path.join(self._tmp_path,f'sim_{self._name}')
        # Files paths
        self._fasta_tmp = os.path.join(self._tmp_path, f'sim_{self._name}_tmp.fasta')
        self._R1_fastq = os.path.join(self._tmp_path, f'sim_{self._name}_R1.fastq')
        self._R2_fastq = os.path.join(self._tmp_path, f'sim_{self._name}_R2.fastq')
        self._fasta_out = os.path.join(outdir, f'sim_{self._name}_data.fna.gz')
        self._cls_out = os.path.join(outdir, f'sim_{self._name}_class.csv')
        # Dataset variables
        self.kmers_data = {}
        os.mkdir(self._tmp_path)

    def simulation(self, k = None, kmers_list = None):
        k, kmers_list = self._verify_sim_arguments(k, kmers_list)
        self._make_tmp_fasta()
        cmd = f"iss generate -g {self._fasta_tmp} -n {self._nb_reads} --abundance halfnormal --model {self._sequencing} --output {self._prefix} --cpus {os.cpu_count()}"
        os.system(cmd)
        self._fastq2fasta()
        self._write_cls_file()
        if k is not None and kmers_list is not None:
            self._kmers_dataset(k, kmers_list)
            generated_files = glob(f'{self._prefix}*')
            for file in generated_files:
                os.remove(file)
            return self.kmers_data
            
    def _make_tmp_fasta(self):
        for file in [self._fasta_in, self._fasta_host]:
            if isinstance(file, Path) or isinstance(file, str):
                if os.path.isfile(file):
                    if os.path.splitext(file)[1] == '.gz':
                        self._add_tmp_fasta_gz(file)
                    else:
                        self._add_tmp_fasta_fa(file)
                elif os.path.isdir(file):
                    self._add_tmp_fasta_dir(file)
            elif isinstance(file, list):
                for f in file:
                    if os.path.splitext(f)[1] == '.gz':
                        self._add_tmp_fasta_gz(f)
                    else:
                        self._add_tmp_fasta_fa(f)

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

    def _add_tmp_fasta_dir(self, dir):
        files_lst = []
        for ext in ['.fa', '.fna', '.fasta','.gz']:
            files_lst.extend(glob(os.path.join(dir, f'*{ext}')))

        with parallel_backend('threading'):
            fastas_to_write = Parallel(n_jobs = -1, prefer = 'threads', verbose = 1)(
                delayed(self._parallel_fasta_to_write)
                (file) for file in files_lst)
        
        with open(self._fasta_tmp, 'at') as handle:
            for record in fastas_to_write:
                SeqIO.write(record, handle, 'fasta')

    def _parallel_fasta_to_write(self, fasta):
        if os.path.splitext(fasta)[1] == '.gz':
            return self._parallel_read_gz(fasta)
        else:
            return self._parallel_read_fa(fasta)
            
    def _parallel_read_gz(self, file):
        with gzip.open(file, 'rt') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                if record.id in self._genomes:
                    return record
    
    def _parallel_read_fa(self, file):
        with open(file, 'rt') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                if record.id in self._genomes:
                    return record

    def _fastq2fasta(self):
        with open(self._R1_fastq, "rt") as handle_R1, open(self._R2_fastq, "rt") as handle_R2, gzip.open(self._fasta_out, "at") as handle_out:
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
            f'simulation_{self._name}',
            kmers_list = kmers_list,
            k = k
        )

    def _verify_sim_arguments(self, k, kmers_list):
        if k is None and kmers_list is not None:
            warn("K-mers list provided but k is None, k will be set to length of k-mers in the list")
            k = len(kmers_list[0])
        elif k is not None and kmers_list is None:
            warn("K is provided but k-mers list is None, k-mers list will be generated")
            raise ValueError("k value was provided but not k-mers list, please provide a k-mers list or no k value")
        return k, kmers_list

# Helper functions
#########################################################################################################

def split_sim_dataset(ds, data, name):
    splitted_path = os.path.join(os.path.dirname(data['profile']), f'Xy_genome_simulation_{name}_data_K{len(data["kmers"][0])}')
    if os.path.exists(splitted_path):
        warnings.warn(f'Splitted dataset {name} already exists, skipping simulation')
        return None
    else:
        splitted_ds = ds.random_sample(0.1)
        if splitted_ds.count() == 0:
            nb_samples = round(ds.count() * 0.1)
            splitted_ds = ds.random_shuffle().limit(nb_samples)
        
        sim_dataset(ds, data, name)
        return splitted_ds

def sim_dataset(ds, data, name):
    """
    Simulate the dataset from the database and generate its data
    """
    k = len(data['kmers'][0])
    cols = ['id']
    cols.extend(data['taxas'])
    cls = pd.DataFrame(columns = cols)
    for batch in ds.iter_batches(batch_format = 'pandas'):
        cls = pd.concat([cls, batch[cols]], axis = 0, ignore_index = True)
    
    sim_outdir = os.path.dirname(data['profile'])
    cv_sim = readsSimulation(data['fasta'], cls, list(cls['id']), 'miseq', sim_outdir, name)
    sim_data = cv_sim.simulation(k, data['kmers'])
    files_lst = glob(os.path.join(sim_data['profile'], '*.parquet'))
    sim_ds = ray.data.read_parquet_bulk(files_lst, parallelism = len(files_lst))
    return sim_ds