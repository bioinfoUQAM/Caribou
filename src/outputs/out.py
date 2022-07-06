import numpy as np
import modin.pandas as pd

from Bio import SeqIO
from copy import copy

import os
import ray
import gzip
import pickle

from subprocess import run

__author__ = 'Nicolas de Montigny'

__all__ = ['Outputs']

class Outputs():
    """
    ----------
    Attributes
    ----------

    dataset : string
        Name of the dataset

    host : string
        Name of the host if there is one

    classified_data : dictionnary
        The classes that were predicted by models

    data_labels : list
        Labels used to classify the sequences

    taxas : list
        Taxa levels that were classified

    order : list
        The order in which the classification was made
        Should be from the most specific to less specific

    ----------
    Methods
    ----------

    abundances : Generates an abundances table in csv format
        No parameters required

    kronagram : Generates a Kronagram (interactive tree) in html format
        No parameters required

    report : Generates a full report on identification of classified sequences
        No parameters required

    fasta : Generates a fasta file containing each sequences assigned to a taxonomy for classification made
        No parameters required

    """
    def __init__(database_kmers, results_dir, k, classifier, dataset, host, classified_data):
        # Third-party path
        self._krona_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'KronaTools','scripts','ImportText.pl')
        self._perl_loc = run('which perl', shell = True, capture_output = True, text = True).stdout.strip('\n')
        # Variables
        self.host = host
        self.dataset = dataset
        self.classified_data = classified_data
        self.taxas = database_kmers['taxas']
        self.order = classified_data['order']
        self.data_labels = database_kmers['classes']
        # File names
        self._abund_file = '{}abundance_K{}_{}_{}.csv'.format(results_dir, k, classifier, dataset)
        self._summary_file = '{}summary_K{}_{}_{}.csv'.format(results_dir, k, classifier, dataset)
        self._krona_file = '{}kronagram_K{}_{}_{}.csv'.format(results_dir, k, classifier, dataset)
        self._krona_out = '{}kronagram_K{}_{}_{}.html'.format(results_dir, k, classifier, dataset)
        self._report_file = '{}full_report_K{}_{}_{}.csv'.format(results_dir, k, classifier, dataset)
        self._fasta_outdir = '{}fasta_by_taxa_k{}_{}_{}'.format(results_dir, k, classifier, dataset)
        # Initialize empty
        self._abundances = {}
        self._summary = {}
        # Get abundances used for other outputs
        self._get_abundances()


    def _get_abundances(self):
        for taxa in self.order:
            df = pd.read_parquet(self.classified_data[taxa]['profile'])
            if taxa in ['bacteria','host','unclassified']:
                self._abundances[taxa] = len(df)
            else:
                self._abundances[taxa] = {}
                for cls in np.unique(df['classes']):
                    if cls in self._abundances[taxa]:
                        self._abundances[taxa][cls] += 1
                    else:
                        self._abundances[taxa][cls] = 1

    def abundances(self):
        self._abundance_table()
        print('Abundance table saved to {}'.format(self._abund_file))
        self._summary['initial'] = len(self.data_labels)
        self._summary_table()
        print('Summary table saved to {}'.format(self._summary_file))

    def _abundance_table(self):
        # Abundance tables / relative abundance
        self._summary['total'] = 0
        cols = ['Taxonomic classification','Number of reads','Relative Abundance (%)']
        nrows = 0
        for taxa in self._abundances:
            if taxa not in ['bacteria','host','unclassified']:
                nrows += 1

        df = pd.DataFrame(np.zeros((nrows,len(cols)), dtype = int), index = np.arange(nrows), columns = cols)

        index = 0
        total_abund = 0

        for taxa in self.order:
            if taxa in ['bacteria','host','unclassified']:
                self._summary[taxa] = self._abundances[taxa]
            else:
                df.loc[index, 'Taxonomic classification'] = taxa
                taxa_ind = copy(index)
                taxa_abund = 0
                index += 1
                for k, v in self._abundances[taxa].items():
                    df.loc[index, 'Taxonomic classification'] = k
                    df.loc[index, 'Number of reads'] = v
                    taxa_abund += v
                    total_abund += v
                    index += 1
                df.loc[taxa_ind, 'Number of reads'] = taxa_abund
                self._summary[taxa] = taxa_abund
        df['Relative Abundance (%)'] = (df['Number of reads']/total_abund)*100
        self._summary['total'] = total_abund
        df.to_csv(self._abund_file, na_rep = '', header = True, index = False)

    def _summary_table(self):
        # Summary file of operations / counts & proportions of reads at each steps
        cols = ['Value']
        rows = ['Abundance','','Number of reads before classification', 'Number of reads classified','Number of unclassified reads','Number of reads identified as bacteria']
        values = np.array([np.NaN, np.NaN, self._summary['initial'], self._summary['total'], self._summary['unclassified'], self._summary['bacteria']])
        if self.host is not None:
            rows.append('Number of reads identified as {}'.format(self.host))
            values = np.append(values, self._summary['host'])
        for taxa in self.order:
            if taxa not in ['bacteria','host','unclassified']:
                rows.append('Number of reads classified at {} level'.format(taxa))
                values = np.append(values, self._summary[taxa])
        rows.extend(['','Relative abundances','','Percentage of reads classified', 'Percentage of reads unclassified','Percentage of reads identified as bacteria'])
        values = np.append(values, [np.NaN,np.NaN,np.NaN,(self._summary['total']/self._summary['initial']*100),(self._summary['unclassified']/self._summary['initial']*100),(self._summary['bacteria']/self._summary['initial']*100)])
        if self.host is not None:
            rows.append('Percentage of reads identified as {}'.format(self.host))
            values = np.append(values, self._summary['host']/self._summary['initial']*100)
        for taxa in self.order:
            if taxa not in ['bacteria','host','unclassified']:
                rows.append('Percentage of reads classified at {} level'.format(taxa))
                values = np.append(values, self._summary[taxa]/self._summary['initial']*100)
        df = pd.DataFrame(values, index = rows, columns = cols)
        df.to_csv(self._summary_file, na_rep = '', header = False, index = True)

    def kronagram(self):
        # Kronagram / interactive tree
        self._create_krona_file()
        cmd = '{} {} {} -o {} -n {}'.format(self._perl_loc, self._krona_path, self._krona_file, self._krona_out, self.dataset)
        run(cmd, shell = True)

    def _create_krona_file(self):
        cols = ['Abundance']
        [cols.append(self.taxas[i]) for i in range(len(self.taxas)-1, -1, -1)]
        nrows = 0
        for taxa in self._abundances:
            if taxa not in ['bacteria','host','unclassified']:
                nrows += len(self._abundances[taxa])

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

    def report(self):
        # Report file of classification of each id
        cols = ['Sequence ID']
        [cols.append(self.taxas[i]) for i in range(len(self.taxas)-1, -1, -1)]
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

    def fasta(self):
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
            taxa_dir = os.path.join(self._fasta_outdir,taxa)
            df = pd.read_parquet(self.classified_data[taxa]['profile'])
            for cls in np.unique(df['classes']):
                outfile_cls = os.path.join(taxa_dir,'{}.fna.gz'.format(cls))
                df_cls = df[df['classes'].str.match(cls)]
                ids = list(df_cls['id'])
                with gzip.open(outfile_cls, 'w') as handle:
                    for id in ids:
                        SeqIO.write(records[id], handle, 'fasta')
