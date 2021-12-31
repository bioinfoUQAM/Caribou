#!/bin/bash

#SBATCH --job-name=generate_db_files
#SBATCH --account=def-kembelst
#SBATCH --mem=191000M
#SBATCH --time=168:00:00
#SBATCH --error=/home/nicdemon/projects/def-kembelst/nicdemon/Maitrise/db/GTDB/%x_%a.err
#SBATCH --output=/home/nicdemon/projects/def-kembelst/nicdemon/Maitrise/db/GTDB/%x_%a.out

FASTA_DIR=/home/nicdemon/projects/def-kembelst/nicdemon/Maitrise/db/GTDB/gtdb_genomes_reps_r202/

LIST_FILES_IN=/home/nicdemon/projects/def-kembelst/nicdemon/Maitrise/db/GTDB/genome_paths.tsv
CLS_FILES_IN=/home/nicdemon/projects/def-kembelst/nicdemon/Maitrise/db/GTDB/gtdb_sp_clusters.tsv

OUT_DIR=/home/nicdemon/projects/def-kembelst/nicdemon/Maitrise/db/GTDB/


#time $(sh $OUT_DIR/generateFastaList.sh -d $FASTA_DIR -o $LIST_FILES_IN)

time $(sh $OUT_DIR/fasta2class_bacteria.sh -d $FASTA_DIR -i $LIST_FILES_IN -c $CLS_FILES_IN -o $OUT_DIR)
