# Caribou
Alignment-free bacterial identification and classification in metagenomics sequencing data using machine learning.

## Installation
The Caribou analysis pipeline was developped in python3 and can be easily installed through the python wheel. The repo must be cloned first and then the package can be installed using the following commands lines in the desired folder :
```
git clone https://github.com/bioinfoUQAM/Caribou.git
pip install path/to/Caribou/
```

### Dependencies
The Caribou analysis pipeline is packed with executables for all dependencies that cannot be installed through the python wheel. These dependencies are:
- [faSplit](https://github.com/ucscGenomeBrowser/kent/blob/8b379e58f89d4a779e768f8c41b042bda714d101/src/utils/faSplit/faSplit.c)
- [KMC](https://github.com/refresh-bio/KMC)
- [KronaTools](https://github.com/marbl/Krona/tree/master/KronaTools)
- [A RAPIDS container](https://rapids.ai/index.html)

### [Optional] GPU acceleration
#### GPU setup
It is recommended to run on a GPU-enabled machine to accelerate training of models but it is not necessary.
To install GPU dependencies on your machine if wanted, refer to following tutorials for installation of GPUs :
- [cudnn](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
- [GPU for tensorflow](https://www.tensorflow.org/install/gpu)
- [CUDA](https://developer.nvidia.com/cuda-downloads)

#### RAPIDS
To accelerate extraction of K-mers profiles by using one or more GPUs, it is necessary to have the RAPIDS library installed. For installation see [RAPIDS installation instructions](https://rapids.ai/start.html#get-rapids).

A Singularity image with the Caribou package already installed can be found in the folder `Caribou/containers`:
- The file `Caribou.sif` is a  modified docker image containing the [RAPIDS library v.22.02 with CUDA 11.5 built on CentOS 8 built by NVIDIA for cloud computing in a conda environment](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/rapidsai/containers/rapidsai).

Should the user want to use this environment, this can be done easily through [Singularity](https://apptainer.org/docs/user/main/index.html) with the following commands :
```
singularity shell --nv -B a/folder/containing/data/to/include/in/the/environment path/to/Caribou/Caribou/supplement/Caribou.sif
```

### [Optional] Python virtual environment
It is recommended to use the analysis pipeline in a virtual environment to be sure that no other installed package can interfere. \
Here is an example of Linux command shell to install Caribou in a new virtual environment by modifying the paths:

```
python3 -m venv /path/to/your/environment/folder/Caribou

source /path/to/your/environment/folder/Caribou/bin/activate

pip install --no-index --upgrade pip

pip install /path/to/Caribou/
```

To access your virtual environment later on the user will only need to run the following two commands:

```
source /path/to/your/environment/folder/Caribou/bin/activate
```

## Building database
Caribou was developed having in mind that the models should be trained on the [GTDB taxonomy database](https://gtdb.ecogenomic.org/). \
Theoritically, any database could be used to train and classify using Caribou but a certain structure should be used for feeding to the program. The specific structure of the database files necessary for training is explained in more details in the [database section of the wiki](https://github.com/bioinfoUQAM/Caribou/wiki/Building-database).

### GTDB pre-extracted K-mers
Extracted K-mers profile files for the GTDB taxonomy representatives version 202 with a k length of 35 can be found in the folder `Caribou/data/kmers`.

### Building GTDB from another release
Should the user want to use a more recent release of the GTDB taxonomy, this can be done using the template script to build data in one large fasta file and extract classes into a csv file. This template must be modified by the user to insert filepaths and comment the host section if there is no host to be used.

The modified template can be submitted to an HPC cluster managed by Slurm (ex: Compute Canada) using the following command :
```
sbatch Caribou/data/build_data_scripts/template_slurm_datagen.sh
```

The modified template can also be ran in a Linux command shell by running the following command :
```
sh Caribou/data/build_data_scripts/template_slurm_datagen.sh
```

Finally each script used by the template can be used alone in Linux command shell by running the following commands :
```
# Generate a list of all fastas to be merged
sh Caribou/data/build_data_scripts/generateFastaList.sh -d [directory] -o [outputFile]

# Extract classes for each bacterial genome fasta using the GTDB taxonomy
sh Caribou/data/build_data_scripts/fasta2class_bact.sh -d [directory] -i [inputFile] -c [classesFile] -o [outputDirectory]

# Extract classes for each host fasta
sh Caribou/data/build_data_scripts/fasta2class_host.sh -d [directory] -i [inputFile] -o [outputDirectory]
```

## Pretrained models
Pretrained models are available in the folder `Caribou/data/models`.

The pretrained models available were trained using the GTDB taxonomy representatives version 202 and can be used directly for extraction of bacteria without a host as well as for bacteria classification of the associated taxonomic levels.
Should the user want to use another database or version of the GTDB taxonomy, there will be a training step which can vary in time and computing ressources needed according to its size and the length of the k-mers used. Moreover, the accuracy of the classification depends greatly on the database used and it is recommended to use the [GTDB representatives in the latest release available](https://data.gtdb.ecogenomic.org/releases/latest).

## Usage
The Caribou analysis pipeline requires only a [configuration file](https://github.com/bioinfoUQAM/Caribou/wiki/Configuration-file) to be executed. \
All the informations required by the program are located in this configuration file and are described in the wiki. \
There is a template config file which can be found here `Caribou/eval_configs/template_config.ini`.

Once the installation is done and the configuration file is ready, the following command can be used to launch the pipeline:
```
Caribou.py -c path/to/your/config.ini
```

## Partial analysis scripts
There are also partial steps scripts that can be used should the user want to.

* K-mers_extract_dataset
> This script extracts K-mers of the given dataset using the available ressources on the computer before saving it to drive.
> The usage is : K-mers_extract_dataset.py [-h] -seq SEQ_FILE -cls CLS_FILE -dt DATASET_NAME -k K_LENGTH [-l KMERS_LIST] -o OUTDIR

* Bacteria_extraction_train_cv
> This script trains and cross-validates a model for the bacteria extraction / host removal step.
> The usage is : Bacteria_extraction_train_cv.py [-h] -db DATA_BACTERIA [-dh DATA_HOST] -dt DATABASE_NAME [-model {None,linearsvm,attention,lstm,deeplstm}] [-bs BATCH_SIZE] [-cv NB_CV_JOBS] [-v] -o OUTDIR

* Bacteria_classification_train_cv
> This script trains and cross-validates a model for the bacteria classification step.
> The usage is : Bacteria_classification_train_cv.py [-h] -db DATA_BACTERIA -dt DATABASE_NAME [-model {ridge,svm,mlr,mnb,lstm_attention,cnn,widecnn}] [-bs BATCH_SIZE] [-cv NB_CV_JOBS] [-v] -o OUTDIR
