#!/usr/bin/env python

from Caribou.data.build_data import *
from Caribou.models.bacteria_extraction import *
from Caribou.models.classification import *

import pandas as pd

from tensorflow.compat.v1 import ConfigProto, Session
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.config import list_physical_devices

import sys
import configparser
import os.path
from os import makedirs

__author__ = "Nicolas de Montigny"

__all__ = []

# GPU & CPU setup
################################################################################
gpus = list_physical_devices('GPU')
if gpus:
    config = ConfigProto(device_count={'GPU': len(gpus), 'CPU': os.cpu_count()})
    sess = Session(config=config)
    set_session(sess);

# Part 0 - Initialisation / extraction of parameters from config file
################################################################################

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Config file is missing ! ! !")
        sys.exit()

    print("Running {}".format(sys.argv[0]), flush=True)

    # Get argument values from ini file
    config_file = sys.argv[1]
    config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())

    with open(config_file, "r") as cf:
        config.read_file(cf)

    # names
    database = config.get("name", "database", fallback = "database")
    metagenome = config.get("name", "metagenome", fallback = "metagenome")
    host = config.get("name", "host", fallback = None)

    # io
    database_seq_file = config.get("io", "database_seq_file")
    database_cls_file = config.get("io", "database_cls_file")
    host_seq_file = config.get("io", "host_seq_file", fallback = None)
    host_cls_file = config.get("io", "host_cls_file", fallback = None)
    metagenome_seq_file = config.get("io", "metagenome_seq_file")
    outdir = config.get("io", "outdir")

    # seq_rep
    k_length = config.getint("seq_rep", "k", fallback = 20)
    fullKmers = config.getboolean("seq_rep", "full_kmers", fallback = True)
    lowVarThreshold = config.getfloat("seq_rep", "low_var_threshold", fallback = None)

    # settings
    binary_classifier = config.get("settings", "host_extractor", fallback = "attention")
    multi_classifier = config.get("settings", "bacteria_classifier", fallback = "lstm_attention")
    cv = config.getboolean("settings", "cross_validation", fallback = True)
    n_cvJobs = config.getint("settings", "nb_cv_jobs", fallback = 1)
    verbose = config.getboolean("settings", "verbose", fallback = True)
    training_batch_size = config.getint("settings", "training_batch_size", fallback = 32)
# AMINE -> IDEAS TO ADAPT SAVING
    binary_saving_host = config.getboolean("settings", "binary_save_host", fallback = True)
    binary_saving_unclassified = config.getboolean("settings", "binary_save_unclassified", fallback = True)
    classifThreshold = config.getfloat("settings", "classification_threshold", fallback = 0.8)

# Part 0.5 - Validation of parameters and environment
################################################################################

    # io
    for file in [database_seq_file, database_cls_file, metagenome_seq_file]:
        if not os.path.isfile(file):
            print("Cannot find file {} ! Exiting".format(file))
            sys.exit()

    if host not in ['none', 'None', None]:
        for file in [host_seq_file, host_cls_file]:
            if not os.path.isfile(file):
                print("Cannot find file {} ! Exiting".format(file))
                sys.exit()

    outdir_path, outdir_folder = os.path.split(outdir)
    if not os.path.isdir(outdir) and os.path.exists(outdir_path):
        print("Created output folder")
        os.makedirs(outdir)
    elif not os.path.exists(outdir_path):
        print("Cannot find output folder ! Exiting")
        sys.exit()

    # seq_rep
    if type(k_length) != int or k_length <= 0:
        print("Invalid kmers length ! Please enter a positive integer ! Exiting")
        print("Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki")
        sys.exit()
    if fullKmers not in [True, False, None]:
        print("Invalid value for full_kmers ! Please use boolean values ! Exiting")
        print("Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki")
        sys.exit()
    if lowVarThreshold is not None:
        if type(lowVarThreshold) == float and not 0 < lowVarThreshold <= 1:
            print("Invalid variance threshold for extracting k-mers ! Please enter a value between 0 and 1 ! Exiting")
            print("Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki")
            sys.exit()

    # settings
    if binary_classifier not in ["onesvm","linearsvm","attention","lstm","deeplstm"]:
        print("Invalid host extraction classifier ! Exiting")
        print("Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki")
        sys.exit()
    if multi_classifier not in ["ridge","svm","mlr","mnb","lstm_attention","cnn","deepcnn"]:
        print("Invalid multiclass bacterial classifier ! Exiting")
        print("Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki")
        sys.exit()
    if cv not in [True, False, None]:
        print("Invalid value for cross_validation ! Please use boolean values ! Exiting")
        print("Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki")
        sys.exit()
    if type(n_cvJobs) != int or n_cvJobs <= 0:
        print("Invalid number of cross validation jobs ! Please enter a positive integer ! Exiting")
        print("Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki")
        sys.exit()
    if verbose not in [True, False, None]:
        print("Invalid value for verbose parameter ! Please use boolean values ! Exiting")
        print("Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki")
        sys.exit()
    if type(training_batch_size) != int or training_batch_size <= 0:
        print("Invalid number of training batch size ! Please enter a positive integer ! Exiting")
        print("Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki")
        sys.exit()
    if binary_saving_host not in [True, False, None]:
        print("Invalid value for host data saving ! Please use boolean values ! Exiting")
        print("Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki")
        sys.exit()
    if binary_saving_unclassified not in [True, False, None]:
        print("Invalid value for unclassifiable sequences ! Please use boolean values ! Exiting")
        print("Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki")
        sys.exit()
    if not 0 < classifThreshold <= 1 or type(classifThreshold) != float:
        print("Invalid confidence threshold for classifying bacterial sequences ! Please enter a value between 0 and 1 ! Exiting")
        print("Please refer to the wiki for further details : https://github.com/bioinfoUQAM/Caribou/wiki")
        sys.exit()

    # Adjust classifier based on host presence or not
    if host in ["none", "None", None]:
        binary_classifier = "onesvm"

    # Check lowVarThreshold
    if lowVarThreshold == "None":
        lowVarThreshold = None
    else:
        lowVarThreshold = float(lowVarThreshold)

    # Check batch_size
    if multi_classifier in ["cnn","deepcnn"] and training_batch_size < 20:
        training_batch_size = 20

    # Tags for prefix out
    if fullKmers:
        tag_kf = "F"
    elif lowVarThreshold:
        tag_kf = "V"
    else:
        tag_kf = "S"

    # Folders creation for output
    outdirs = {}
    outdirs["main_outdir"] = os.path.join(outdir, metagenome)
    outdirs["data_dir"] = os.path.join(outdirs["main_outdir"], "data")
    outdirs["models_dir"] = os.path.join(outdirs["main_outdir"], "models")
    outdirs["prefix"] = tag_kf
    makedirs(outdirs["main_outdir"], mode=0o700, exist_ok=True)
    makedirs(outdirs["data_dir"], mode=0o700, exist_ok=True)
    makedirs(outdirs["models_dir"], mode=0o700, exist_ok=True)
    outdirs["data_dir"] = os.path.join(outdirs["data_dir"], outdirs["prefix"])
    outdirs["models_dir"] = os.path.join(outdirs["models_dir"], outdirs["prefix"])

    if cv:
        outdirs["plots_dir"] = os.path.join(outdirs["main_outdir"], "plots")
        makedirs(outdirs["plots_dir"], mode=0o700, exist_ok=True)
        outdirs["plots_dir"] = os.path.join(outdirs["plots_dir"], outdirs["prefix"])

# Part 1 - K-mers profile extraction
################################################################################

    if host != "none":
        # Reference Database and Host
        k_profile_database, k_profile_host = build_load_save_data((database_seq_file, database_cls_file),
            (host_seq_file, host_cls_file),
            outdirs["data_dir"],
            database,
            k = k_length,
            full_kmers = fullKmers,
            low_var_threshold = lowVarThreshold
        )
    else:
        # Reference Database Only
        k_profile_database = build_load_save_data((database_seq_file, database_cls_file),
            host,
            outdirs["data_dir"],
            database,
            k = k_length,
            full_kmers = fullKmers,
            low_var_threshold = lowVarThreshold
        )

    # Metagenome to analyse
    k_profile_metagenome = build_load_save_data(metagenome_seq_file,
        "none",
        outdirs["data_dir"],
        metagenome,
        kmers_list = k_profile_database["kmers_list"]
    )

# Part 2 - Binary classification of bacteria / host sequences
################################################################################

    for binary_classifier in ["linearsvm","attention","lstm","deeplstm"]:
        if host == "none":
            bacterial_metagenome = bacteria_extraction(k_profile_metagenome,
                k_profile_database,
                k_length,
                outdirs,
                database,
                classifier = binary_classifier,
                batch_size = training_batch_size,
                verbose = verbose,
                cv = cv,
                saving_host = binary_saving_host,
                saving_unclassified = binary_saving_unclassified,
                n_jobs = n_cvJobs
                )
        else:
            bacterial_metagenome = bacteria_extraction(k_profile_metagenome,
                (k_profile_database, k_profile_host),
                k_length,
                outdirs,
                database,
                classifier = binary_classifier,
                batch_size = training_batch_size,
                verbose = verbose,
                cv = cv,
                saving_host = binary_saving_host,
                saving_unclassified = binary_saving_unclassified,
                n_jobs = n_cvJobs
                )

# Part 3 - Multiclass classification of bacterial sequences
################################################################################

    for multi_classifier in ["ridge","svm","mlr","mnb","lstm_attention","cnn","deepcnn"]:
        classification_data = bacterial_classification(bacterial_metagenome,
            k_profile_database,
            k_length,
            outdirs,
            database,
            classifier = multi_classifier,
            batch_size = training_batch_size,
            threshold = classifThreshold,
            verbose = verbose,
            cv = cv,
            n_jobs = n_cvJobs)

# Part 5 - Classification refinement
################################################################################
    """
# convert identification en np.array/list/dict de nb reads pr chaque sp
    classification = merge_classified_data(classification_data)

    classif_abundances = classification_abundance(classification)
    """
# dimension reduction for reclassification?
# order of kmers for better signature ~ markov chains

# Part 6 - (OPTIONAL) New sequences identification / clustering
################################################################################

    # Clustering for unidentified sequences into MAGs -> try to assign to species afterward
    # Amine faire attention à comment fait la classif
    """
    from sklearn.cluster import MiniBatchKMeans
        classifier == "kmeans":
            if verbose:
                print("Training multiclass classifier with K Means")
            clf = MiniBatchKMeans(nclusters = nb_classes, batch_size = batch_size, random_state = 42)
    """

# Part 7 - Outputs for biological analysis of bacterial population
################################################################################
    # Kronagram
    # Abundance tables / relative abundance
        # Identification of each sequence \w domain + probability -> cutoff pr user if needed
        # Taxonomic tree / table -> newick
        # Joint identification of reads  vx autres domaines?
    # Option for file containing kmers
    # Summary file of opperations / proportions of reads at each steps
    # Github wiki manual

    # Environment
        # R wrapper / execution in Rmarkdown?
        # Venv / Conda
        # Docker / singularity
