# metagenomics_ML
Alignment-free bacterial identification and classification in metagenomics sequencing data using machine learning

## Installation of dependencies
To install gpu on your machine if necessary, refer to tensorflow's help on installing gpus : https://www.tensorflow.org/install/gpu

## Testing data
Mock community used for testing found in database Mockrobiota from Kozich et al.

## Building database
There is a template script to build data in one large fasta file and extract classes into a csv file.
This template must be modified by the user to insert filepaths and comment the host section if there is no host to be used.

The modified template can be submited on a HPC cluster managed by Slurm (ex: Compute Canada) using the following command :
```
sbatch metagenomics_ML/data/build_data_scripts/template_slurm_datagen.sh
```

The modified template can also be ran in a linux command shell by running the following command :
```
sh metagenomics_ML/data/build_data_scripts/template_slurm_datagen.sh
```

Finally each script used by the template can be used in linux command shell by running the following commands :
```
sh metagenomics_ML/data/build_data_scripts/generateFastaList.sh -d [directory] -o [outputFile]

sh metagenomics_ML/data/build_data_scripts/fasta2class_bact.sh -d [directory] -i [inputFile] -c [classesFile] -o [outputDirectory]

sh metagenomics_ML/data/build_data_scripts/fasta2class_host.sh -d [directory] -i [inputFile] -o [outputDirectory]
```

## Usage
Not in a package yet.

There is a template config file which can be found here `metagenomics_ML/eval_configs/template_config.ini`.

To test, run the following command using you own modified config file :

```
python3 metagenomics_ML/metagenomics_ML/main.py metagenomics_ML/eval_configs/test.ini
```

Description for each variable can be found in the wiki https://github.com/bioinfoUQAM/metagenomics_ML.wiki.git
