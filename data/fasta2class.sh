#!/bin/bash

#SBATCH --job-name=fasta2class_gtdb
#SBATCH --account=def-kembelst
#SBATCH --mem=191000M
#SBATCH --time=24:00:00
#SBATCH --error=/home/nicdemon/projects/def-kembelst/nicdemon/Maitrise/db/GTDB/%x_%a.err
#SBATCH --output=/home/nicdemon/projects/def-kembelst/nicdemon/Maitrise/db/GTDB/%x_%a.out

module load python/3.7
module load imkl/2020.1.217

source /home/nicdemon/projects/def-kembelst/nicdemon/Article_mlr/env_mlr/bin/activate

python3 /home/nicdemon/projects/def-kembelst/nicdemon/Maitrise/db/GTDB/fasta2class.py
