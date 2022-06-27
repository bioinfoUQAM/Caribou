import numpy as np
import modin.pandas as pd

from Bio import SeqIO
from copy import copy

import os
import ray
import gzip
import pickle

from subprocess import run
from utils import load_Xy_data

__author__ = 'Nicolas de Montigny'

__all__ = ['Outputs']

class Outputs():
    def __init__(database_kmers, results_dir, k, classifier, dataset, host, classified_data, seq_file, abundance_stats = True, kronagram = True, full_report = True, extract_fasta = True):
        # Third-party path
        self.krona_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'KronaTools','scripts','ImportText.pl')
        self.perl_loc = run('which perl', shell = True, capture_output = True, text = True).stdout.strip('\n')
        # Variables
        self.database = database_kmers
        self.dataset = dataset
        self.host = host
        self.classified_data = classified_data
        self.order = classified_data['order'].copy()
        with open(seq_file, 'rb') as handle:
            self.data_labels = pickle.load(handle).labels # seq_data.labels
        # File names
        self._abund_file = '{}abundance_K{}_{}_{}.csv'.format(results_dir, k, classifier, dataset)
        self._summary_file = '{}summary_K{}_{}_{}.csv'.format(results_dir, k, classifier, dataset)
        self._krona_file = '{}kronagram_K{}_{}_{}.csv'.format(results_dir, k, classifier, dataset)
        self._krona_out = '{}kronagram_K{}_{}_{}.html'.format(results_dir, k, classifier, dataset)
        self._report_file = '{}full_report_K{}_{}_{}.csv'.format(results_dir, k, classifier, dataset)
        self._fasta_outdir = '{}fasta_by_taxa_k{}_{}_{}'.format(results_dir, k, classifier, dataset)
        # Initialize empty
        self.abundances = {}
        self.summary = {}
        # Get abundances used for other outputs
        self._get_abundances()
        # Output desired files according to parameters
        if abundance_stats is True:
            self._abundances()
        if kronagram is True:
            self._kronagram()
        if full_report is True:
            self._report()
        if extract_fasta is True:
            self._fasta()

    def _get_abundances(self):
        for taxa in self.order:
            df = pd.read_parquet(self.classified_data[taxa]['profile'])
            if taxa in ['bacteria','host','unclassified']:
                self.abundances[taxa] = len(df)
            else:
                self.abundances[taxa] = {}
                for cls in np.unique(df['classes']):
                    if cls in self.abundances[taxa]:
                        self.abundances[taxa][cls] += 1
                    else:
                        self.abundances[taxa][cls] = 1

    def _abundances(self):
        self._abundance_table()
        self.summary['initial'] = len(self.data_labels)
        self._summary()

    def _abundance_table(self):
        # Abundance tables / relative abundance
        self.summary['total'] = 0
        cols = ['Taxonomic classification','Number of reads','Relative Abundance (%)']
        nrows = 0
        for taxa in self.abundances:
            if taxa not in ['bacteria','host','unclassified']:
                nrows += 1

        df = pd.DataFrame(np.zeros((nrows,len(cols)), dtype = int), index = np.arange(nrows), columns = cols)

        index = 0
        total_abund = 0

        for taxa in self.order:
            if taxa in ['bacteria','host','unclassified']:
                self.summary[taxa] = self.abundances[taxa]
            else:
                df.loc[index, 'Taxonomic classification'] = taxa
                taxa_ind = copy(index)
                taxa_abund = 0
                index += 1
                for k, v in self.abundances[taxa].items():
                    df.loc[index, 'Taxonomic classification'] = k
                    df.loc[index, 'Number of reads'] = v
                    taxa_abund += v
                    total_abund += v
                    index += 1
                df.loc[taxa_ind, 'Number of reads'] = taxa_abund
                self.summary[taxa] = taxa_abund
        df['Relative Abundance (%)'] = (df['Number of reads']/total_abund)*100
        self.summary['total'] = total_abund
        df.to_csv(self._abund_file, na_rep = '', header = True, index = False)

    def _summary(self):
        # Summary file of operations / counts & proportions of reads at each steps
        cols = ['Value']
        rows = ['Abundance','','Number of reads before classification', 'Number of reads classified','Number of unclassified reads','Number of reads identified as bacteria']
        values = np.array([np.NaN, np.NaN, self.summary['initial'], self.summary['total'], self.summary['unclassified'], self.summary['bacteria']])
        if self.host is not None:
            rows.append('Number of reads identified as {}'.format(self.host))
            values = np.append(values, self.summary['host'])
        for taxa in self.order:
            if taxa not in ['bacteria','host','unclassified']:
                rows.append('Number of reads classified at {} level'.format(taxa))
                values = np.append(values, self.summary[taxa])
        rows.extend(['','Relative abundances','','Percentage of reads classified', 'Percentage of reads unclassified','Percentage of reads identified as bacteria'])
        values = np.append(values, [np.NaN,np.NaN,np.NaN,(self.summary['total']/self.summary['initial']*100),(self.summary['unclassified']/self.summary['initial']*100),(self.summary['bacteria']/self.summary['initial']*100)])
        if self.host is not None:
            rows.append('Percentage of reads identified as {}'.format(self.host))
            values = np.append(values, self.summary['host']/self.summary['initial']*100)
        for taxa in self.order:
            if taxa not in ['bacteria','host','unclassified']:
                rows.append('Percentage of reads classified at {} level'.format(taxa))
                values = np.append(values, self.summary[taxa]/self.summary['initial']*100)
        df = pd.DataFrame(values, index = rows, columns = cols)
        df.to_csv(self._summary_file, na_rep = '', header = False, index = True)

    def _kronagram(self):
        # Kronagram / interactive tree
        self._create_krona_file()
        cmd = '{} {} {} -o {} -n {}'.format(self.perl_loc, self.krona_path, self._krona_file, self._krona_out, self.dataset)
        run(cmd, shell = True)

    def _create_krona_file(self):
        cols = ['Abundance']
        [cols.append(self.database['taxas'][i]) for i in range(len(self.database['taxas'])-1, -1, -1)]
        nrows = 0
        for taxa in self.abundances:
            if taxa not in ['bacteria','host','unclassified']:
                nrows += len(self.abundances[taxa])

        df = pd.DataFrame(np.zeros((nrows,len(cols)), dtype = int), index = np.arange(nrows), columns = cols)

        unique_rows = np.vstack(list({tuple(row) for row in self.data_labels}))

        index = 0
        for taxa in self.order:
            if taxa not in ['bacteria','host','unclassified']:
                col_taxa = df.columns.get_loc(taxa)
                for k, v in abundances[taxa].items():
                    if k in df[taxa]:
                        ind = df[df[taxa] == k].index.values
                        df.loc[ind, taxa] = k
                        df.loc[ind, 'Abundance'] += v
                        if col_taxa != 1:
                            for col in range(1, col_taxa+1):
                                df.iloc[ind,col] = np.flip(unique_rows[np.where(unique_rows == k)[0]])[0][col-1]
                    else:
                        df.loc[index, taxa] = k
                        df.loc[index, 'Abundance'] += v
                        if col_taxa != 1:
                            for col in range(1, col_taxa+1):
                                df.iloc[index,col] = np.flip(unique_rows[np.where(unique_rows == k)[0]])[0][col-1]
                        index += 1
        df = df.fillna(0)
        df.to_csv(self._krona_file, na_rep = '', header = False, index = False)

    def _report(self):
        # Report file of classification of each id
        cols = ['Sequence ID']
        [cols.append(self.database_kmers['taxas'][i]) for i in range(len(self.database_kmers['taxas'])-1, -1, -1)]
        nrows = 0
        for taxa in self.order:
            if taxa not in ['bacteria','host','unclassified']:
                nrows += len(self.classified_data[taxa]['ids'])

        df = pd.DataFrame(np.zeros((nrows,len(cols)), dtype = int), index = np.arange(nrows), columns = cols)

        unique_rows = np.vstack(list({tuple(row) for row in self.data_labels}))

        index = 0
        for taxa in self.order:
            if taxa not in ['bacteria','host','unclassified']:
                col_taxa = df.columns.get_loc(taxa)
                for id, classification in zip(self.classified_data[taxa]['ids'], self.classified_data[taxa]['classification']):
                    df.loc[index, 'Sequence ID'] = id
                    df.loc[index, taxa] = classification
                    if col_taxa != 1:
                        for col in range(1, col_taxa+1):
                            df.iloc[index,col] = np.flip(unique_rows[np.where(unique_rows == classification)[0]])[0][col-1]
                    index += 1

        df = df.fillna(0)
        df.to_csv(self._report_file, na_rep = '', header = True, index = False)

    def _fasta(self):
        os.mkdir(self._fasta_outdir)
        path, ext = os.path.splitext(self.fasta_file)
        if ext == '.gz':
            with gzip.open(self.fasta_file, 'rt') as handle:
                records = SeqIO.index(handle, 'fasta')
        else:
            with open(self.fasta_file, 'rt') as handle:
                records = SeqIO.index(handle, 'fasta')

        list_taxa = [order[i] for i in range(len(order)-1, -1, -1)]
        list_taxa.remove('unclassified')



    for taxa in list_taxa:
        taxa_dir = os.path.join(_fasta_outdir,taxa)
        df = vaex.open(classified_data[taxa]['profile'])
        for cls in df.unique('classes'):
            outfile_cls = os.path.join(taxa_dir,'{}.fa.gz'.format(cls))
            df_cls = df[df.classes.str.match(cls)]
            ids = list(df_cls.id.values)
            with gzip.open(outfile_cls, 'w') as handle:
                for id in ids:
                    SeqIO.write(records[id], handle, 'fasta')
