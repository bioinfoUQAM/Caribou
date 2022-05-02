#!/bin/bash

__author__="Nicolas de Montigny"

HELP=0
while getopts "d:i:c:o:h" option; do
  case "${option}" in
    i) FASTA=${OPTARG};;
    input) FASTA=${OPTARG};;
    c) CLASSES_IN=${OPTARG};;
    classes) CLASSES_IN=${OPTARG};;
    o) OUTFILE=${OPTARG};;
    output) OUTFILE=${OPTARG};;
    h) HELP=1;;
    help) HELP=1;;
  esac
done

if [ $HELP -eq 1 ]; then
  """
  usage : fasta2class_mini.sh -i [inputFile] -c [classesFile] -o [outputDirectory]

  This script subsets the class file to correspond to the fasta file of a subset of a database.
  To create a subset of a fasta file beforehand, 'seqtk sample' can be used.

  -i --input A fasta file containing a subset of the database
  -c --classes A csv containing all classes of the database
  -o --output File where the subset classes will be written
  -h --help Show this help message
  """
  exit 0
fi

echo "id","species","genus","family","order","class","phylum","domain" >> $OUTFILE

if [[ $FASTA == *.gz ]]; then
  list_ids=$(zcat $FASTA | grep -o -E "^>\w+" | tr -d ">")
else
  list_ids=$(grep -o -E "^>\w+" $FASTA | tr -d ">")
fi

for id in $list_ids; do
  cat $CLASSES_IN | grep $id >> $OUTFILE
done
