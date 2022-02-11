
import numpy as np
import pandas as pd
import tables as tb

from subprocess import run

from copy import copy

import pickle

import os

from Caribou.utils import load_Xy_data

__author__ = "Nicolas de Montigny"

__all__ = ["outputs","get_abundances","out_abundances","abundance_table","out_summary","out_kronagram","create_krona_file","out_report"]

def outputs(database_kmers, results_dir, k, classifier, dataset, host, classified_data, seq_file, abundance_stats = True, kronagram = True, full_report = True):
    abund_file = "{}abundance_K{}_{}_{}.tsv".format(results_dir, k, classifier, dataset)
    summary_file = "{}summary_K{}_{}_{}.tsv".format(results_dir, k, classifier, dataset)
    krona_file = "{}kronagram_K{}_{}_{}.tsv".format(results_dir, k, classifier, dataset)
    krona_out = "{}kronagram_K{}_{}_{}.html".format(results_dir, k, classifier, dataset)
    report_file = "{}full_report_K{}_{}_{}.tsv".format(results_dir, k, classifier, dataset)
    tree_file = "{}taxonomic_tree_K{}_{}_{}.nwk".format(results_dir, k, classifier, dataset)

    with open(seq_file, "rb") as handle:
        seq_data = pickle.load(handle)

    abundances, order = get_abundances(classified_data)

    if abundance_stats is True:
        out_abundances(abundances, order, abund_file, summary_file, host, seq_data)
    if kronagram is True:
        out_kronagram(abundances, order, krona_file, krona_out, seq_data, database_kmers, dataset)
    if full_report is True:
        out_report(classified_data, order, report_file, database_kmers, seq_data)

def get_abundances(data):
    abundances = {}
    order = data["order"].copy()

    for taxon in data["order"]:
        if taxon in ["bacteria","host","unclassified"]:
            abundances[taxon] = len(data[taxon]["ids"])
        else:
            abundances[taxon] = {}
            for classif in data[taxon]["classification"]:
                if classif in abundances[taxon]:
                    abundances[taxon][classif] += 1
                else:
                    abundances[taxon][classif] = 1

    return abundances, order

def out_abundances(abundances, order, abund_file, summary_file, host, seq_data):
    summary = abundance_table(abundances, order, abund_file)
    summary["initial"] = len(seq_data.labels)
    out_summary(abundances, order, summary, host, summary_file)

def abundance_table(abundances, order, abund_file):
    # Abundance tables / relative abundance
    summary = {"total":0}
    cols = ["Taxonomic classification","Number of reads","Relative Abundance (%)"]
    nrows = 0
    for taxon in abundances:
        if taxon not in ["bacteria","host","unclassified"]:
            nrows += 1

    df = pd.DataFrame(np.zeros((nrows,len(cols)), dtype = int), index = np.arange(nrows), columns = cols)

    index = 0
    total_abund = 0
    for taxon in order:
        if taxon in ["bacteria","host","unclassified"]:
            summary[taxon] = copy(abundances[taxon])
        else:
            df.loc[index, "Taxonomic classification"] = taxon
            taxon_ind = copy(index)
            taxon_abund = 0
            index += 1
            for k, v in abundances[taxon].items():
                df.loc[index, "Taxonomic classification"] = k
                df.loc[index, "Number of reads"] = v
                taxon_abund += v
                total_abund += v
                index += 1
            df.loc[taxon_ind, "Number of reads"] = taxon_abund
            summary[taxon] = taxon_abund
    df["Relative Abundance (%)"] = (df["Number of reads"]/total_abund)*100
    summary["total"] = total_abund
    df.to_csv(abund_file, sep = "\t", na_rep = "", header = True, index = False)

    return summary

def out_summary(abundances, order, summary, host, summary_file):
    # Summary file of operations / counts & proportions of reads at each steps
    cols = ["Value"]
    rows = ["Abundance","","Number of reads before classification", "Number of reads classified","Number of unclassified reads","Number of reads identified as bacteria"]
    values = np.array([np.NaN, np.NaN, summary["initial"], summary["total"], summary["unclassified"], summary["bacteria"]])
    if host is not None:
        rows.append("Number of reads identified as {}".format(host))
        values = np.append(values, summary["host"])
    for taxon in order:
        if taxon not in ["bacteria","host","unclassified"]:
            rows.append("Number of reads classified at {} level".format(taxon))
            values = np.append(values, summary[taxon])
    for i in ["","Relative abundances","","Percentage of reads classified", "Percentage of reads unclassified","Percentage of reads identified as bacteria"]:
        rows.append(i)
    values = np.append(values, [np.NaN,np.NaN,np.NaN,(summary["total"]/summary["initial"]*100),(summary["unclassified"]/summary["initial"]*100),(summary["bacteria"]/summary["initial"]*100)])
    if host is not None:
        rows.append("Percentage of reads identified as {}".format(host))
        values = np.append(values, summary["host"]/summary["initial"]*100)
    for taxon in order:
        if taxon not in ["bacteria","host","unclassified"]:
            rows.append("Percentage of reads classified at {} level".format(taxon))
            values = np.append(values, summary[taxon]/summary["initial"]*100)

    df = pd.DataFrame(values, index = rows, columns = cols)
    df.to_csv(summary_file, sep = "\t", na_rep = "", header = False, index = True)

def out_kronagram(abundances, order, krona_file, krona_out, seq_data, database_kmers, dataset):
    # Kronagram / interactive tree
    krona_path = "{}/KronaTools/scripts/ImportText.pl".format(os.path.dirname(os.path.realpath(__file__)))
    create_krona_file(abundances, order, krona_file, seq_data, database_kmers)
    perl_loc = run("which perl", shell = True, capture_output = True, text = True)
    cmd = "{} {} {} -o {} -n {}".format(perl_loc.stdout.strip("\n"), krona_path, krona_file, krona_out, dataset)
    run(cmd, shell = True)

def create_krona_file(abundances, order, krona_file, seq_data, database_kmers):
    cols = ['Abundance']
    [cols.append(database_kmers['taxas'][i]) for i in range(len(database_kmers['taxas'])-1, -1, -1)]

    nrows = 0
    for taxon in abundances:
        if taxon not in ["bacteria","host","unclassified"]:
            nrows += len(abundances[taxon])

    df = pd.DataFrame(np.zeros((nrows,len(cols)), dtype = int), index = np.arange(nrows), columns = cols)

    unique_rows = np.vstack(list({tuple(row) for row in seq_data.labels}))

    index = 0
    for taxon in order:
        if taxon not in ["bacteria","host","unclassified"]:
            col_taxon = df.columns.get_loc(taxon)
            for k, v in abundances[taxon].items():
                if k in df[taxon]:
                    ind = df[df[taxon] == k].index.values
                    df.loc[ind, taxon] = k
                    df.loc[ind, "Abundance"] += v
                    if col_taxon != 1:
                        for col in range(1, col_taxon+1):
                            df.iloc[ind,col] = np.flip(unique_rows[np.where(unique_rows == k)[0]])[0][col-1]
                else:
                    df.loc[index, taxon] = k
                    df.loc[index, "Abundance"] += v
                    if col_taxon != 1:
                        for col in range(1, col_taxon+1):
                            df.iloc[index,col] = np.flip(unique_rows[np.where(unique_rows == k)[0]])[0][col-1]
                    index += 1
    df = df.replace(0,np.NaN)
    df.to_csv(krona_file, sep = "\t", na_rep = "", header = False, index = False)

def out_report(classified_data, order, report_file, database_kmers, seq_data):
    # Report file of classification of each id
    cols = ['Sequence ID']
    [cols.append(database_kmers['taxas'][i]) for i in range(len(database_kmers['taxas'])-1, -1, -1)]

    nrows = 0
    for taxon in order:
        if taxon not in ["bacteria","host","unclassified"]:
            nrows += len(classified_data[taxon]["ids"])

    df = pd.DataFrame(np.zeros((nrows,len(cols)), dtype = int), index = np.arange(nrows), columns = cols)

    unique_rows = np.vstack(list({tuple(row) for row in seq_data.labels}))

    index = 0
    for taxon in order:
        if taxon not in ["bacteria","host","unclassified"]:
            col_taxon = df.columns.get_loc(taxon)
            for id, classification in zip(classified_data[taxon]["ids"], classified_data[taxon]["classification"]):
                df.loc[index, "Sequence ID"] = id
                df.loc[index, taxon] = classification
                if col_taxon != 1:
                    for col in range(1, col_taxon+1):
                        df.iloc[index,col] = np.flip(unique_rows[np.where(unique_rows == classification)[0]])[0][col-1]
                index += 1
    df = df.replace(0,np.NaN)
    df.to_csv(report_file, sep = "\t", na_rep = "", header = True, index = False)
