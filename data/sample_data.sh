#!/bin/bash

__author__="Nicolas de Montigny"

HELP=0
while getopts f:p:o:hfastapercentoutputhelp option; do
  case "${option}" in
    f) FILE=${OPTARG};;
    fasta) FILE=${OPTARG};;
    p) PERCENT=${OPTARG};;
    percent) PERCENT=${OPTARG};;
    o) OUTFILE=${OPTARG};;
    output) OUTFILE=${OPTARG};;
    h) HELP=1;;
    help) HELP=1;;
  esac
done

if [ $HELP -eq 1 ];
then
  """
  usage : sample_data.sh -f [fasta] -p [percent] -t [tmp] -o [output]

  This script extract a given percentage picked at random from the fasta file given in input.
  This was used to accelerate the testing of the Caribou pipeline and to get preliminary results before extracting all the GTDB database.

  -f --fasta Path to a fasta file containing multiple sequences from which to extract a percentage of sequences
  -p --percent Percentage of sequences to extract from the fasta file
  -o --output Path to output file where the extracted sequences will be stored
  -h --help Show this message
  """
fi

# Make tmp dir
DIR=$( realpath $FILE  )
DIR=$( dirname "$DIR" )
TMP=$DIR/tmp

mkdir $TMP

# Split fasta by sequences
../Caribou/data/faSplit sequence $FILE 1000000000 $TMP/fasta

# List of files in tmp
for file in $TMP/*; do
  echo $file >> $TMP/fileList.txt
done

fileList=$TMP/fileList.txt

length=$(wc -l < $TMP/fileList.txt)

nbSeq=$(expr $PERCENT \* $length / 100)

range=$(shuf -i 1-$length -n$nbSeq)

for i in $range; do
  sequence=$(sed -n "${i}p" $fileList)
  cat $sequence >> $OUTFILE
done

rm -r $TMP
