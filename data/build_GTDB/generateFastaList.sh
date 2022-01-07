#!/bin/bash

__author__="Nicolas de Montigny"

HELP=0
while getopts d:o:hdirectoryoutputhelp option; do
  case "${option}" in
    d) DIR=${OPTARG};;
    directory) DIR=${OPTARG};;
    o) OUTFILE=${OPTARG};;
    output) OUTFILE=${OPTARG};;
    h) HELP=1;;
    help) HELP=1;;
  esac
done

if [ $HELP -eq 1 ];
then
  """
  usage : generateFastaList.sh -d [directory] -o [outputFile]

  This script generates the list of files in all subdirectories up to depth 10 of a given directory.
  This list will be saved into output file given by user.

  -d --directory A directory containing all fasta files and possibly subdirs
  -o --output Path to output file
  -h --help Show this help message
  """
fi

for i in $(find $DIR -maxdepth 10 -type f); do
  echo $i >> $OUTFILE
done
