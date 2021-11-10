#!/bin/bash

#SBATCH --job-name=generate_db_files
#SBATCH --account=[accountname]
#SBATCH --mem=[memory required / wanted]
#SBATCH --time=[time to run]
#SBATCH --error=[path/to/log/outdirectory]/%x_%a.err
#SBATCH --output=[path/to/log/outdir]/%x_%a.out

"""
Template script to submit database merging and class generation on HPC clusters which uses Slurm Workload Manager
Otherwise each script can be used in linux command line in itself

User need only to change paths to their own and change path to other scripts if launching from elsewhere

If there is no host to extract sequences and classes from, simply comment the lines for the host
"""

FASTA_DIR_BACT=[path/to/directory/containing/subdirs/or/fastas/to/merge/for/bacterias]

LIST_FILES_BACT=[path/to/file/containing/path/to/files/to/be/merged/for/bacterias]
CLS_FILE_BACT=[path/to/file/containing/information/of/classes/related/to/sequences/in/fasta/files]

OUT_DIR=[path/to/directory/where/outputs/will/be/writen]

sh $SLURM_SUBMIT_DIR/generateFastaList.sh -d $FASTA_DIR_BACT -o $LIST_FILES_BACT

sh $SLURM_SUBMIT_DIR/fasta2class_bacteria.sh -d $FASTA_DIR_BACT -i $LIST_FILES_BACT -c $CLS_FILE_BACT -o $OUTDIR



# Comment this section if there is no host sequence using # at the beginning of each lines
FASTA_DIR_HOST=[path/to/directory/containing/subdirs/or/fastas/to/merge/for/host/if/there/is/one]

LIST_FILES_HOST=[path/to/file/containing/path/to/files/to/be/merged/for/host/if/there/is/one]

sh $SLURM_SUBMIT_DIR/generateFastaList.sh -d $FASTA_DIR_HOST -o $LIST_FILES_HOST

sh $SLURM_SUBMIT_DIR/fasta2class_host.sh -d $FASTA_DIR_HOST -i $LIST_FILES_HOST -o $OUT_DIR
