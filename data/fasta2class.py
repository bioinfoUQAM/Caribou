from Bio import SeqIO
import gzip
import pandas as pd

fasta_dir = "/home/nicdemon/projects/def-kembelst/nicdemon/Maitrise/db/GTDB/gtdb_genomes_reps_r202/"

list_files = "/home/nicdemon/projects/def-kembelst/nicdemon/Maitrise/db/GTDB/genome_paths.tsv"
gtdb_cls_files = "/home/nicdemon/projects/def-kembelst/nicdemon/Maitrise/db/GTDB/gtdb_sp_clusters.tsv"

fasta = "/home/nicdemon/projects/def-kembelst/nicdemon/Maitrise/db/GTDB/data.fa.gz"
cls = "/home/nicdemon/projects/def-kembelst/nicdemon/Maitrise/db/GTDB/class.csv"

list_ids = []

gtdb_cls = pd.read_csv(gtdb_cls_files, sep = "\t", header = 0, index_col = 0)
file_paths = pd.read_csv(list_files, sep = " ", header = None)
classes = pd.DataFrame(columns = ["id","species","genus","family","order","class","phylum","domain"])

for i in range(len(file_paths)):
    file = file_paths.iloc[i,0]
    path = file_paths.iloc[i,1]
    with gzip.open("{}/{}/{}".format(fasta_dir,path,file), "rt") as fasta_in, gzip.open(fasta, "at") as fasta_out:
        records = list(SeqIO.parse(fasta_in, "fasta"))
        list_ids = pd.DataFrame({0 : [record.id for record in records]})
        SeqIO.write(records, fasta_out, "fasta")
        for i in range(len(list_ids)):
            tmp = file.split(" ")[0].split(".")
            gb = "{}.{}".format(tmp[0], tmp[1].split("_")[0])
            gb_row = pd.DataFrame(gtdb_cls.loc[gb]).transpose()
            gb_row.insert(0, "id", list_ids.iloc[i,0])
            classes = classes.append(gb_row)

with open(cls, "wt") as cls_out:
    classes.to_csv(cls_out, header = True, index = True)
