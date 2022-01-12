# Caribou
Alignment-free bacterial identification and classification in metagenomics sequencing data using machine learning

## Installation of dependencies
It is recommended to run on a GPU-enabled machine to accelerate training of models but not necessary.
To install GPU dependencies on your machine if wanted, refer to following tutorials for installation of GPUs :
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html \
https://www.tensorflow.org/install/gpu

To install package in a new python virtual environment through pip, make sure that python is installed first.
Modify the first line to choose where to pu your virtual environment on your drive.
In linux command shell, execute the following commands :

```
ENV_DIR=/path/to/your/environment/folder

python3 -m venv $ENV_DIR/Caribou

source $ENV_DIR/Caribou/bin/activate

pip install --no-index --upgrade pip

pip install /home/nicolas/github/Caribou
```

To access your virtual environment later on you will only need to run the following two commands

```
ENV_DIR=/path/to/your/environment/folder

source $ENV_DIR/Caribou/bin/activate
```

## Building database
There is a template script to build data in one large fasta file and extract classes into a csv file.
This template must be modified by the user to insert filepaths and comment the host section if there is no host to be used.

The modified template can be submited on a HPC cluster managed by Slurm (ex: Compute Canada) using the following command :
```
sbatch Caribou/data/build_data_scripts/template_slurm_datagen.sh
```

The modified template can also be ran in a linux command shell by running the following command :
```
sh Caribou/data/build_data_scripts/template_slurm_datagen.sh
```

Finally each script used by the template can be used in linux command shell by running the following commands :
```
sh Caribou/data/build_data_scripts/generateFastaList.sh -d [directory] -o [outputFile]

sh Caribou/data/build_data_scripts/fasta2class_bact.sh -d [directory] -i [inputFile] -c [classesFile] -o [outputDirectory]

sh Caribou/data/build_data_scripts/fasta2class_host.sh -d [directory] -i [inputFile] -o [outputDirectory]
```

## Pretrained models
Pretrained models are available in the folder

The pretrained models available were trained using the GTDB taxonomy representatives version 202 and can be used directly for both extraction of bacteria with (squash sequences) and without host as well as for bacteria classification of the associated taxonomic levels.
Should the user want to use another database or version of the GTDB taxonomy, there will be a training step which can vary in time and computing ressources needed according to its size and the length of the k-mers used. Moreover, the accuracy of the classification depends greatly on the database used and it is recommended to use the GTDB representatives in the latest release available [here](https://data.gtdb.ecogenomic.org/releases/).

## Usage
The package can be easily installed through pip. The git must be cloned first and then the package can be installed using the following commands lines in the desired folder :
```
git clone https://github.com/bioinfoUQAM/Caribou.git
pip install ./Caribou/
```

There is a template config file which can be found here `Caribou/eval_configs/template_config.ini`.

To use, run the following command using you own modified config file :

```
main.py Caribou/eval_configs/test.ini
```

Description for each variable can be found in the [wiki](https://github.com/bioinfoUQAM/Caribou/wiki/Configuration-file).
