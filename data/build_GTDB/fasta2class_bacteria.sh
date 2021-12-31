#!/bin/bash

__author__="Nicolas de Montigny"

HELP=0
while getopts d:i:c:o:h option; do
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
  usage : fasta2class_bacteria.sh -d [directory] -i [inputFile] -c [classesFile] -o [outputDirectory]

  This script merges multiple fasta files into one and creates the file containing classes of those sequences
  This method was tested on GTDB database and taxonomy and might need some modifications for other taxonomies

  -d --directory a directory containing all fasta files
  -i --input a tsv/csv file containing path and names to all fasta files to extract ids from
  -c --classes A tsv/csv
  -o --output Path to output files
  -h --help Show this help message
  """
fi



fasta_file=$OUTDIR/data_bacteria.fa.gz
tmp_file=$OUTDIR/tmp.csv
cls_file=$OUTDIR/class_bacteria.csv
echo "id","species","genus","family","order","class","phylum","domain" >> $cls_file
length=$(wc -l $FASTA_LIST | awk '{print $1}')

for i in $(seq $length); do
  echo $i:$length
  declare -a list_ids=()
  file=$(sed -n "${i}p" $FASTA_LIST)
  cat $file >> $fasta_file
  GCA=$(basename $file | cut -d'_' -f1-2)
  list_ids+=$(zcat $file | grep ">" | awk '{print $1}' | sed 's/>//')
  for id in ${list_ids[*]}; do
    entry=$(cat $CLASSES_IN | grep $GCA)
    entry=$(echo ${entry/$GCA/$id})
    echo $entry >> $tmp_file
  done
done
awk ' {print $1 "," $2 " " $3 "," $4 "," $5 "," $6 "," $7 "," $8 "," $9}' $tmp_file >> $cls_file
rm $tmp_file
