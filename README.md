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

### [Recommanded] Containers
Containers with the Caribou package and all dependencies already installed can be found in the folder `Caribou/containers`.
It is recommended to execute the Caribou pipeline inside a container to ease usage and reproductibility.

The Caribou containers are modified versions of Tensorflow containers and require to have the following dependencies installed prior to use:
- [Docker](https://www.docker.com/) or [Singularity / Apptainer](https://apptainer.org/docs/user/main/index.html) depending on the version used.
- [NVIDIA GPU Drivers](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) which are platform specific.
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

#### Docker
[Docker](https://www.docker.com/) is a container virtual environment that can be run on any desktop or cloud.
It can be installed on any system following the [instructions provided in the Docker documentation](https://www.docker.com/get-started/).

The Docker container is built on top of the [official Tensorflow container available on docker hub](https://hub.docker.com/r/tensorflow/tensorflow) with dependencies and Caribou installed in a python virtual environment.

It must be imported from the tar archive included in this repo before being able to use it.
Assuming that docker is already installed in command line :
```
docker load path/to/Caribou/containers/caribou_docker.tar
```

After having imported the container, it can be used in two ways:
- Shell : Use the environment in a shell interface and run commands as described later in this document.
```
docker run --gpus all -it caribou_docker
```
- Process : Execute  in the background commands described later in this document. Replace [command] with the script that should be executed followed by it's options.
```
docker run --gpus all -d caribou_docker  [command].py
```

#### Singularity / Apptainer
[Singularity / Apptainer](https://apptainer.org/docs/user/main/index.html) is an HPC optimized container system and should be used instead of Docker if the user wants to use a container on an HPC cluster.
It is often already installed on HPC clusters, but should the user need to install it, [instructions can be found here.](https://apptainer.org/docs/user/main/quick_start.html)

The Singularity / apptainer container is a [Tensorflow container optimised for HPC by NVIDIA](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) with dependencies and Caribou installed in a python virtual environment.

As with docker, the environment can be used in two ways :
- Shell : Use the environment in a shell interface and run commands as described later in this document.

```
singularity shell --nv -B a/folder/containing/data/to/bind/in/the/environment path/to/Caribou/containers/Caribou_singularity.sif
```
- Process : Execute instructions scripts when using the container for production on a compute cluster managed by schedulers (ex: Slurm, Torque, PBS, etc.). Instructions for usage of the exec command are provided in the [documentation.](https://apptainer.org/docs/user/main/cli/apptainer_exec.html) and applied example on Compute Canada clusters using Slurm Workload Manager can be found on their [wiki.](https://docs.computecanada.ca/wiki/Singularity#Running_a_single_command). Usage may differ slightly depending on the HPC clusters and Workload Managers used.

### [Optional] GPU acceleration
Usage of machine learning models can be accelerated by using a GPU but it is not necessary.
If using a container, these dependencies are already installed. Otherwise they should be installed prior to analysis accelerated by GPU.

To install GPU dependencies on your machine, refer to following tutorials for installation :
- [CUDA](https://developer.nvidia.com/cuda-downloads)
- [cudnn](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
- [GPU for tensorflow](https://www.tensorflow.org/install/gpu)

### [Optional] Python virtual environment
If not using a container, it is recommended to use the analysis pipeline in a virtual environment to be sure that no other installed package can interfere. \
Here is an example of Unix-like command shell to install Caribou in a new virtual environment by modifying the paths:

```
python3 -m venv /path/to/your/environment/folder

source /path/to/your/environment/folder/bin/activate

pip install --no-index --upgrade pip

pip install /path/to/downloaded/Caribou/folder
```

To access your virtual environment later on the user will only need to run the following two commands:

```
source /path/to/your/environment/folder/bin/activate
```

## Building database
Caribou was developed having in mind that the models should be trained on the [GTDB taxonomy database](https://gtdb.ecogenomic.org/). \
Theoritically, any database could be used to train and classify using Caribou but a certain structure should be used for feeding to the program. The specific structure of the database files necessary for training is explained in more details in the [database section of the wiki](https://github.com/bioinfoUQAM/Caribou/wiki/Building-database).

### GTDB pre-extracted K-mers
Extracted K-mers profile files for the [GTDB representatives version 202](https://data.gtdb.ecogenomic.org/releases/release202/202.0/) with a k length of 20 can be found in the folder `Caribou/data/kmers`.

### Building GTDB from another release
Should the user want to use a more recent release of the GTDB taxonomy, this can be done using the template script to build data in one large fasta file and extract classes into a csv file. This template must be modified by the user to insert filepaths and comment the host section if there is no host to be used.

The modified template can be submitted to an HPC cluster managed by Slurm (ex: Compute Canada) using the following command :
```
sbatch Caribou/data/build_data_scripts/template_slurm_datagen.sh
```

The modified template can also be ran in a Unix-like command shell by running the following command :
```
sh Caribou/data/build_data_scripts/template_slurm_datagen.sh
```

Finally each script used by the template can be used alone in Unix-like command shell by running the following commands :
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
Caribou_pipeline.py -c path/to/your/config.ini
```

## Partial analysis scripts
There are also partial steps scripts that can be used should the user want to.

* Caribou_pipeline.py
> This script runs the entire Caribou analysis Pipeline
> Usage : Caribou_pipeline.py [-c CONFIG_FILE]

* Caribou_kmers.py
> This script extracts K-mers of the given dataset using the available ressources on the computer before saving it to drive. \
> usage: Caribou_kmers.py [-h] [-s SEQ_FILE] [-c CLS_FILE] [-dt DATASET_NAME] [-sh SEQ_FILE_HOST] [-ch CLS_FILE_HOST] [-dh HOST_NAME] -k K_LENGTH [-l KMERS_LIST] -o OUTDIR

* Caribou_extraction.py
> This script trains a model and extracts bacteria / host sequences. \
> usage: Caribou_extraction.py [-h] -db DATA_BACTERIA [-dh DATA_HOST] -mg DATA_METAGENOME -dt DATABASE_NAME [-ds HOST_NAME] -mn METAGENOME_NAME [-model {None,onesvm,linearsvm,attention,lstm,deeplstm}] [-bs BATCH_SIZE] [-e TRAINING_EPOCHS] [-v] -o OUTDIR [-wd WORKDIR]

* Caribou_classification.py
> This script trains a model and classifies bacteria sequences iteratively over known taxonomic levels. \
> usage: Caribou_classification.py [-h] -db DATA_BACTERIA -mg DATA_METAGENOME -dt DATABASE_NAME -mn METAGENOME_NAME [-model {sgd,mnb,lstm_attention,cnn,widecnn}] [-t TAXA] [-bs BATCH_SIZE] [-e TRAINING_EPOCHS] [-v] -o OUTDIR [-wd WORKDIR]

* Caribou_outputs.py
> This script produces outputs from the results of classified data by Caribou. \
> usage: Caribou_outputs.py [-h] -db DATA_BACTERIA -clf CLASSIFIED_DATA -model {sgd,mnb,lstm_attention,cnn,widecnn} -dt DATASET_NAME [-ds HOST_NAME] [-a] [-k] [-r] [-f]

* Caribou_extraction_train_cv.py
> This script trains and cross-validates a model for the bacteria extraction / host removal step. \
> usage: Caribou_extraction_train_cv.py [-h] -db DATA_BACTERIA [-dh DATA_HOST] -dt DATABASE_NAME [-ds HOST_NAME] [-model {None,onesvm,linearsvm,attention,lstm,deeplstm}] [-bs BATCH_SIZE] [-e TRAINING_EPOCHS] [-v] -o OUTDIR [-wd WORKDIR]

* Caribou_classification_train_cv.py
> This script trains and cross-validates a model for the bacteria classification step. \
> usage: Caribou_classification_train_cv.py [-h] -db DATA_BACTERIA -dt DATABASE_NAME [-model {sgd,mnb,lstm_attention,cnn,widecnn}] [-bs BATCH_SIZE] [-e TRAINING_EPOCHS] [-v] -o OUTDIR [-wd WORKDIR]
