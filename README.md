# Caribou
Alignment-free bacterial identification and classification in metagenomics sequencing data using machine learning.

## Concept
A recurrent problem in metagenomics studies is the lack of valorization of the data collected when analysing it, as ~30-50% can be classified to a taxonomy. Depending on the methods and algorithms used, it can also be a fastidious method that requires a lot of memory and/or time to compute.

Caribou was designed having these problems in mind. Therefore, it leverages machine learning algorithms and neural networks to classify metagenomics sequencing data to multiple taxonomic levels without requiring alignment methods.

To do so, the workflow consists of 4 steps:
1. K-mers representation of genetic sequences
2. Bacterial sequences identification or host sequences exclusion
3. Top-down bacterial sequences classification
4. Output to user

![image](https://user-images.githubusercontent.com/61320660/156250025-f7c08bd6-b15e-4e5c-9949-e9f1afe56296.png)
> [Proof of concept is supplied in the wiki](https://github.com/bioinfoUQAM/Caribou/wiki/Proof-of-concept) should users want to see how the Caribou package works

## Data

**Analysis**

To use the Caribou pipeline for analysis, only a fasta file is required. It can either be gzipped or raw.

**Training**

The models used by the pipeline can be retrained should the user want to use more specialized datasets or a newer version of the database.
To do so, a new database should be built as described in the [*building database* section of the wiki](https://github.com/bioinfoUQAM/Caribou/wiki/Building-database).

**Data used for training & benchmarking**

The models used by Caribou in steps 2 & 3 were trained on a curated version of the [GTDB v.202 representatives](https://data.gtdb.ecogenomic.org/releases/release202/202.0/).
A benchmark was also executed to compare the performances of the Caribou pipeline with state-of-the-art tools.

The data used for training and benchmarking the models is further described in the [*training & benchmark datasets* section of the wiki](https://github.com/bioinfoUQAM/Caribou/wiki/Training-&-benchmark-datasets).

## Installation
The Caribou pipeline is built using Python3 and requires some dependencies to be installed.

For further information on how to install the package ses the [*installation* section of the wiki](https://github.com/bioinfoUQAM/Caribou/wiki/Installation).

## Usage
The Caribou pipeline can be executed in it's entirety by supplying a config file from the linux command line.

Each step can also be executed by itself by using the corresponding scripts from the linux command line.

More details on scripts usage and options are discussed in the [*usage* section of the wiki](https://github.com/bioinfoUQAM/Caribou/wiki/Usage).