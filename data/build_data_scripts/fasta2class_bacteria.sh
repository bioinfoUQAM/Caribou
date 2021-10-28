#!/bin/bash

HELP=0
while getopts d:i:c:o:hdirectoryinputclassesoutputhelp option; do
  case "${option}" in
    d) DIR=${OPTARG};;
    directory) DIR=${OPTARG};;
    i) FASTA_LIST=${OPTARG};;
    input) FASTA_LIST=${OPTARG};;
    c) CLASSES_IN=${OPTARG};;
    classes) CLASSES_IN=${OPTARG};;
    o) OUTDIR=${OPTARG};;
    output) OUTDIR=${OPTARG};;
    h) HELP=1;;
    help) HELP=1;;
  esac
done

if [ $HELP -eq 1 ];
then
  """
  usage : fasta2class_bact.sh -d [directory] -i [inputFile] -c [classesFile] -o [outputDirectory]

  This script merges multiple fasta files into one and creates the file containing classes of those sequences
  This method was tested on GTDB database and taxonomy and might need some modifications for other taxonomies

  -d --directory a directory containing all fasta files
  -i --input a tsv/csv file containing path and names to all fasta files to extract ids from
  -c --classes A tsv/csv
  -o --output Path to output files
  -h --help Show this help message
  """
fi

declare -a list_ids=()
#ARRAY_NAME+=(NEW_ITEM1)

fasta_file=$DIR/data_bacteria.fa.gz
cls_file=$DIR/class_bacteria.csv
echo "id","species","genus","family","order","class","phylum","domain" >> $cls_file

for i in $(seq $(wc -c $FASTA_LIST | awk '{print $1}')); do
  file=$(sed -n "${i}p" $FASTA_LIST)
  cat $file >> $fasta_file
  ids=$(zcat $file | grep ">" | awk '{print $1}') | sed 's/>//'
  for j in $(seq $(zcat $file | grep -c ">")); do
    id=$(sed -n "${j}p" $ids)
    echo $(cat $CLASSES_IN | grep $id) >> $cls_file
  done
done
