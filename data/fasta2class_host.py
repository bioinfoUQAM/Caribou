from Bio import SeqIO
import gzip
import pandas as pd
from sys import argv

__author__ = "nicolas"

"""
usage : python3 fasta2class_host.py fasta_file_input.fa.gz class_file_output.csv host_species
"""

fasta_input = argv[1]

cls = argv[2]

host_species = argv[3]

list_ids = []

classes = pd.DataFrame(columns = ["id","species","domain"])

with gzip.open(fasta_input, "rt") as fasta_in:
    records = list(SeqIO.parse(fasta_in, "fasta"))
    list_ids = pd.DataFrame({"id" : [record.id for record in records]})
    for i in range(len(list_ids)):
        classes = classes.append({"id" : str(list_ids.iloc[i,0]), "species" : str(host_species), "domain" : "host"}, ignore_index = True)

with open(cls, "wt") as cls_out:
    classes.to_csv(cls_out, header = True, index = False)
