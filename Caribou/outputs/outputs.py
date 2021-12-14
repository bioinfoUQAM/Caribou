
import numpy as np
import pandas as pd
import tables as tb

from subprocess import run

import pickle

import os

from Caribou.utils import load_Xy_data

__author__ = "Nicolas de Montigny"

__all__ = ['outputs','out_kronagram','out_abundance_table','out_summary']

#Functions
def outputs(database_kmers, outdirs, k, classifier, dataset, classified_data, seq_data, abundance_table = True, kronagram = True, stats = True, taxo_tree = True, fasta = True):
    if abundance_table is True:
        out_abundance_table()
    if kronagram is True:
        out_kronagram(outdirs, k, classifier, dataset, classified_data, seq_data, database_kmers)
    if stats is True:
        out_summary()
    if taxo_tree is True:
        out_tree()
    if fasta is True:
        out_fasta()
"""
classified_data = {}
classified_data["X"] = str(classified_kmers_file)
classified_data["kmers_list"] = kmers_list
classified_data["ids"] = [ids[i] for i in classified]
classified_data["classification"] = from_int_cls(classified, ids, predict, labels_list_str)
"""

def out_abundance_table():
    # Abundance tables / relative abundance
        # Identification of each sequence \w domain + probability
        # Joint identification of reads vx autres domaines?
    print("To do")

def out_kronagram(outdirs, k, classifier, dataset, classified_data, seq_data, database_kmers):
    krona_path = "{}/KronaTools/scripts/ImportText.pl".format(os.path.dirname(os.path.realpath(__file__)))
    print(krona_path)
    # Kronagram
    krona_file = "{}/K{}_{}_kronagram_{}.txt".format(outdirs["data_dir"], k, classifier, dataset)
    krona_out = "{}/K{}_{}_kronagram_{}.html".format(outdirs["plots_dir"], k, classifier, dataset)
    create_krona_file(krona_file, classified_data, seq_data, database_kmers)
    perl_loc = run("which perl", shell = True, capture_output = True, text = True)
    cmd = "{} {} {} -o {} -n {}".format(perl_loc.stdout.strip("\n"), krona_path, krona_file, krona_out, dataset)
    run(cmd, shell = True)

def create_krona_file(file, data, seq_data, database_kmers):
    abundances = {}
    col = ['Abundance']
    [col.append(database_kmers['taxas'][i]) for i in range(len(database_kmers['taxas'])-1, -1, -1)]

    for taxon, data_tax in data.items():
        abundances[taxon] = {}
        for classif in data_tax["classification"]:
            if classif in abundances[taxon]:
                abundances[taxon][classif] += 1
            else:
                abundances[taxon][classif] = 1

    abund_array = np.array([abundances[x] for x in abundances]).T
    length = 0
    for tax in abundances:
        length += len(abundances[tax])

    df = pd.DataFrame(np.zeros((length,len(col)), dtype = int), index = np.arange(length), columns = col)

    unique_rows = np.vstack(list({tuple(row) for row in seq_data.labels}))

    index = 0
    for tax in abundances:
        col_tax = df.columns.get_loc(tax)
        for k, v in abundances[tax].items():
            if k in df[tax]:
                ind = df[df[tax] == k].index.values
                df.loc[ind, tax] = k
                df.loc[ind, "Abundance"] += v
                if col_tax != 1:
                    for col in range(1, col_tax+1):
                        df.iloc[ind,col] = np.flip(unique_rows[np.where(unique_rows == k)[0]])[0][col-1]
            else:
                df.loc[index, tax] = k
                df.loc[index, "Abundance"] += v
                if col_tax != 1:
                    for col in range(1, col_tax+1):
                        df.iloc[index,col] = np.flip(unique_rows[np.where(unique_rows == k)[0]])[0][col-1]
                index += 1
    df = df.replace(0,np.NaN)
    df.to_csv(file, sep = "\t", na_rep = "", header = False, index = False, )


def out_summary():
    # Summary file of operations / stats / proportions of reads at each steps
    print("To do")

def out_tree():
    # Taxonomic tree / table -> newick format
    print("To do")

def out_fasta():
    # Fasta file of classifications made
    print("To do")

#Testing
seqfile = "/home/nicolas/github/Caribou_exp/local/results/output/mock/data/S_seqdata_db_bacteria_DB.txt"
with open(seqfile, "rb") as handle:
    seq_data = pickle.load(handle)
database_kmers = load_Xy_data("/home/nicolas/github/Caribou_exp/local/results/output/mock/data/S_K4_Xy_genome_bacteria_DB_data.npz")
outdirs = {"data_dir":"/home/nicolas/github/Caribou_exp/local/results/output/mock/data/", "plots_dir":"/home/nicolas/github/Caribou_exp/local/results/output/mock/plots/"}
classified_data = {"species":{"X":"path","kmers_list":["ACTG","AGTC","GTCA"],"ids":["1","2","3"],"classification":["test_1","test_2","test_3"]}, "domain":{"X":"path","kmers_list":["ACTG","AGTC","GTCA"],"ids":["1","2","3"],"classification":["bacteria","bacteria","-1"]}}
outputs(database_kmers, outdirs, 4, "TEST", "Patate", classified_data, seq_data)
