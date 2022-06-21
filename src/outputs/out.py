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

__all__ = ['to_user','get_abundances','out_abundances','abundance_table','out_summary','out_kronagram','create_krona_file','out_report','out_fasta']

def to_user(database_kmers, results_dir, k, classifier, dataset, host, classified_data, seq_file, input_fasta_file, abundance_stats = True, kronagram = True, full_report = True, extract_fasta = True):
    abund_file = '{}abundance_K{}_{}_{}.csv'.format(results_dir, k, classifier, dataset)
    summary_file = '{}summary_K{}_{}_{}.csv'.format(results_dir, k, classifier, dataset)
    krona_file = '{}kronagram_K{}_{}_{}.csv'.format(results_dir, k, classifier, dataset)
    krona_out = '{}kronagram_K{}_{}_{}.html'.format(results_dir, k, classifier, dataset)
    report_file = '{}full_report_K{}_{}_{}.csv'.format(results_dir, k, classifier, dataset)
    tree_file = '{}Taxonomic_tree_K{}_{}_{}.nwk'.format(results_dir, k, classifier, dataset)
    fasta_outdir = '{}fasta_by_taxa_k{}_{}_{}'.format(results_dir, k, classifier, dataset)

    os.mkdir(fasta_outdir)

    with open(seq_file, 'rb') as handle:
        seq_data = pickle.load(handle)

    abundances, order = get_abundances(classified_data)

    if abundance_stats is True:
        out_abundances(abundances, order, abund_file, summary_file, host, seq_data)
    if kronagram is True:
        out_kronagram(abundances, order, krona_file, krona_out, seq_data, database_kmers, dataset)
    if full_report is True:
        out_report(classified_data, order, report_file, database_kmers, seq_data)
    if extract_fasta is True:
        out_fasta(classified_data, order, fasta_file, fasta_outdir)

def get_abundances(data):
    abundances = {}
    order = data['order'].copy()

    for taxa in order:
        df = ray.data.read_parquet(data[taxa]['profile']).to_modin()
        if taxa in ['bacteria','host','unclassified']:
            abundances[taxa] = len(df)
        else:
            abundances[taxa] = {}
            for cls in np.unique(df['classes']):
                if cls in abundances[taxa]:
                    abundances[taxa][cls] += 1
                else:
                    abundances[taxa][cls] = 1

    return abundances, order

def out_abundances(abundances, order, abund_file, summary_file, host, seq_data):
    summary = abundance_table(abundances, order, abund_file)
    summary['initial'] = len(seq_data.labels)
    out_summary(abundances, order, summary, host, summary_file)

def abundance_table(abundances, order, abund_file):
    # Abundance tables / relative abundance
    summary = {'total':0}
    cols = ['Taxonomic classification','Number of reads','Relative Abundance (%)']
    nrows = 0
    for taxa in abundances:
        if taxa not in ['bacteria','host','unclassified']:
            nrows += 1

    df = pd.DataFrame(np.zeros((nrows,len(cols)), dtype = int), index = np.arange(nrows), columns = cols)

    index = 0
    total_abund = 0
    for taxa in order:
        if taxa in ['bacteria','host','unclassified']:
            summary[taxa] = copy(abundances[taxa])
        else:
            df.loc[index, 'Taxonomic classification'] = taxa
            taxa_ind = copy(index)
            taxa_abund = 0
            index += 1
            for k, v in abundances[taxa].items():
                df.loc[index, 'Taxonomic classification'] = k
                df.loc[index, 'Number of reads'] = v
                taxa_abund += v
                total_abund += v
                index += 1
            df.loc[taxa_ind, 'Number of reads'] = taxa_abund
            summary[taxa] = taxa_abund
    df['Relative Abundance (%)'] = (df['Number of reads']/total_abund)*100
    summary['total'] = total_abund
    df.to_csv(abund_file, na_rep = '', header = True, index = False)

    return summary

def out_summary(abundances, order, summary, host, summary_file):
    # Summary file of operations / counts & proportions of reads at each steps
    cols = ['Value']
    rows = ['Abundance','','Number of reads before classification', 'Number of reads classified','Number of unclassified reads','Number of reads identified as bacteria']
    values = np.array([np.NaN, np.NaN, summary['initial'], summary['total'], summary['unclassified'], summary['bacteria']])
    if host is not None:
        rows.append('Number of reads identified as {}'.format(host))
        values = np.append(values, summary['host'])
    for taxa in order:
        if taxa not in ['bacteria','host','unclassified']:
            rows.append('Number of reads classified at {} level'.format(taxa))
            values = np.append(values, summary[taxa])
    rows.extend(['','Relative abundances','','Percentage of reads classified', 'Percentage of reads unclassified','Percentage of reads identified as bacteria'])
    values = np.append(values, [np.NaN,np.NaN,np.NaN,(summary['total']/summary['initial']*100),(summary['unclassified']/summary['initial']*100),(summary['bacteria']/summary['initial']*100)])
    if host is not None:
        rows.append('Percentage of reads identified as {}'.format(host))
        values = np.append(values, summary['host']/summary['initial']*100)
    for taxa in order:
        if taxa not in ['bacteria','host','unclassified']:
            rows.append('Percentage of reads classified at {} level'.format(taxa))
            values = np.append(values, summary[taxa]/summary['initial']*100)

    df = pd.DataFrame(values, index = rows, columns = cols)
    df.to_csv(summary_file, na_rep = '', header = False, index = True)

def out_kronagram(abundances, order, krona_file, krona_out, seq_data, database_kmers, dataset):
    # Kronagram / interactive tree
    krona_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'KronaTools','scripts','ImportText.pl')
    create_krona_file(abundances, order, krona_file, seq_data, database_kmers)
    perl_loc = run('which perl', shell = True, capture_output = True, text = True)
    cmd = '{} {} {} -o {} -n {}'.format(perl_loc.stdout.strip('\n'), krona_path, krona_file, krona_out, dataset)
    run(cmd, shell = True)

def create_krona_file(abundances, order, krona_file, seq_data, database_kmers):
    cols = ['Abundance']
    [cols.append(database_kmers['taxas'][i]) for i in range(len(database_kmers['taxas'])-1, -1, -1)]

    nrows = 0
    for taxa in abundances:
        if taxa not in ['bacteria','host','unclassified']:
            nrows += len(abundances[taxa])

    df = pd.DataFrame(np.zeros((nrows,len(cols)), dtype = int), index = np.arange(nrows), columns = cols)

    unique_rows = np.vstack(list({tuple(row) for row in seq_data.labels}))

    index = 0
    for taxa in order:
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
    df = df.replace(0,np.NaN)
    df.to_csv(krona_file, na_rep = '', header = False, index = False)

def out_report(classified_data, order, report_file, database_kmers, seq_data):
    # Report file of classification of each id
    cols = ['Sequence ID']
    [cols.append(database_kmers['taxas'][i]) for i in range(len(database_kmers['taxas'])-1, -1, -1)]

    nrows = 0
    for taxa in order:
        if taxa not in ['bacteria','host','unclassified']:
            nrows += len(classified_data[taxa]['ids'])

    df = pd.DataFrame(np.zeros((nrows,len(cols)), dtype = int), index = np.arange(nrows), columns = cols)

    unique_rows = np.vstack(list({tuple(row) for row in seq_data.labels}))

    index = 0
    for taxa in order:
        if taxa not in ['bacteria','host','unclassified']:
            col_taxa = df.columns.get_loc(taxa)
            for id, classification in zip(classified_data[taxa]['ids'], classified_data[taxa]['classification']):
                df.loc[index, 'Sequence ID'] = id
                df.loc[index, taxa] = classification
                if col_taxa != 1:
                    for col in range(1, col_taxa+1):
                        df.iloc[index,col] = np.flip(unique_rows[np.where(unique_rows == classification)[0]])[0][col-1]
                index += 1
    df = df.replace(0,np.NaN)
    df.to_csv(report_file, na_rep = '', header = True, index = False)

def out_fasta(classified_data, order, fasta_file, fasta_outdir):
    path, ext = os.path.splitext(fasta_file)
    if ext == '.gz':
        with gzip.open(fasta_file, 'rt') as handle:
            records = SeqIO.index(handle, 'fasta')
    else:
        with open(fasta_file, 'rt') as handle:
            records = SeqIO.index(handle, 'fasta')

    list_taxa = [order[i] for i in range(len(order)-1, -1, -1)]
    list_taxa.remove('unclassified')
    for taxa in list_taxa:
        taxa_dir = os.path.join(fasta_outdir,taxa)
        df = vaex.open(classified_data[taxa]['profile'])
        for cls in df.unique('classes'):
            outfile_cls = os.path.join(taxa_dir,'{}.fa.gz'.format(cls))
            df_cls = df[df.classes.str.match(cls)]
            ids = list(df_cls.id.values)
            with gzip.open(outfile_cls, 'w') as handle:
                for id in ids:
                    SeqIO.write(records[id], handle, 'fasta')
