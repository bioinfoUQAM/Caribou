#!/bin/bash

__author__="Nicolas de Montigny"

HELP=0
while getopts d:i:o:h option; do
  case "${option}" in
    d) DIR=${OPTARG};;
    directory) DIR=${OPTARG};;
    i) FASTA_LIST=${OPTARG};;
    input) FASTA_LIST=${OPTARG};;
    #g) SPECIES=${OPTARG};;
    #genus) SPECIES=${OPTARG};;
    o) OUTDIR=${OPTARG};;
    output) OUTDIR=${OPTARG};;
    h) HELP=1;;
    help) HELP=1;;
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
  -g --genus name of the host species
  -o --output Path to output directory
  -h --help Show this help message
  """
fi

fasta_file=$OUTDIR/data.fa
cls_file=$OUTDIR/class.csv
echo $OUTDIR
echo $cls_file
echo "id","species","domain" >> $cls_file

for i in $(seq $(wc -l $FASTA_LIST | awk '{print $1}')); do
  file=$(sed -n "${i}p" $FASTA_LIST)
  cat $file >> $fasta_file
  ids=$(cat $file | grep ">" | awk '{print $1}') | sed 's/>//'
  for j in $(seq $(zcat $file | grep -c ">")); do
    id=$(sed -n "${j}p" $ids)
    echo "$id,$SPECIES,host" >> $cls_file
  done
done
