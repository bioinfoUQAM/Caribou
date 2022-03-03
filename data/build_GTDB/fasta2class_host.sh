#!/bin/bash

__author__="Nicolas de Montigny"

HELP=0
while getopts ':d:i:s:o:h' option; do
  case ${option} in
    d) DIR=$OPTARG;;
    i) FASTA_LIST=$OPTARG;;
    s) SPECIES=$OPTARG;;
    o) OUTDIR=$OPTARG;;
    h) HELP=1;;
  esac
done

if [ $HELP -eq 1 ];
then
  """
  usage : fasta2class_host.sh -d [directory] -i [inputFile] -s [species] -o [outputDirectory]

  This script merges multiple fasta files into one and creates the file containing classes of those sequences for a host
  This method was tested on the Cucurbita genre from the NCBI genome datasets

  -d --directory a directory containing all host fasta files
  -i --input a tsv/csv file containing path and names to all fasta files to extract ids from
  -s --species name of the host species
  -o --output Path to output directory
  -h --help Show this help message
  """
  exit 0
fi

fasta_file=$OUTDIR/data.fa
cls_file=$OUTDIR/class.csv
echo "id","species","domain" >> $cls_file

for i in $(seq $(wc -l $FASTA_LIST | awk '{print $1}')); do
  file=$(sed -n "${i}p" $FASTA_LIST)
  cat $file >> $fasta_file
done

list_ids=$(grep -o -E "^>\w+" $fasta_file | tr -d ">")

for id in $list_ids; do
  echo "$id,$SPECIES,host" >> $cls_file
done
