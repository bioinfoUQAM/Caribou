#!/usr/bin/env python

from data.build_data import *
from models.bacteria_extraction import *
from models.classification import *

import pandas as pd

from tensorflow.compat.v1 import ConfigProto, Session
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.config import list_physical_devices

import sys
import configparser
import os.path
from os import makedirs

__author__ = "nicolas"

# GPU & CPU setup
################################################################################
gpus = list_physical_devices('GPU')
if gpus:
    config = ConfigProto(device_count={'GPU': len(gpus), 'CPU': os.cpu_count()})
    sess = Session(config=config)
    set_session(sess);

# Part 0 - Initialisation / extraction of parameters from config file
################################################################################

# CHANGE PARAMETERS TO BE MORE PRECISE / INSTINCTIVE IN CONFIG FILE

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
    database = config.get("name", "database")
    metagenome = config.get("name", "metagenome")
    host = config.get("name", "host")

    # io
    database_seq_file = config.get("io", "database_seq_file")
    database_cls_file = config.get("io", "database_cls_file")
    host_seq_file = config.get("io", "host_seq_file")
    host_cls_file = config.get("io", "host_cls_file")
    metagenome_seq_file = config.get("io", "metagenome_seq_file")
    outdir = config.get("io", "outdir")

    # seq_rep
    # main evaluation parameters
    k_length = config.getint("seq_rep", "k")
    fullKmers = config.getboolean("seq_rep", "full_kmers")
    lowVarThreshold = config.get("seq_rep", "low_var_threshold", fallback=None)

# MAYBE SOME NOT NEEDED
    # evaluation
    cv_folds = config.getint("evaluation", "cv_folds")
    eval_metric = config.get("evaluation", "eval_metric")
    avrg_metric = config.get("evaluation", "avrg_metric")

    # choose classifier based on host presence or not
    if host == "none":
        bacteria_classifier = "onesvm"
    else:
         bacteria_classifier = "linearsvm"

# MAYBE SOME NOT NEEDED
    # settings
    n_mainJobs = config.getint("settings", "n_main_jobs")
    n_cvJobs = config.getint("settings", "n_cv_jobs")
    verbose = config.getint("settings", "verbose")
    training_batch_size = config.getint("settings", "training_batch_size")

    bacteria_saving_host = config.get("settings", "binary_save_host")
    bacteria_saving_unclassified = config.get("settings", "binary_save_unclassified")
    bacteria_cv = config.getint("settings", "binary_cross_val")

# Amine -> ideas to adapt saving
    saveData = config.getboolean("settings", "save_data")
    saveModels = config.getboolean("settings", "save_models")
    saveResults = config.getboolean("settings", "save_results")
    plotResults = config.getboolean("settings", "plot_results")
    randomState = config.getint("settings", "random_state")

    # Check lowVarThreshold
    if lowVarThreshold == "None":
        lowVarThreshold = None
    else:
        lowVarThreshold = float(lowVarThreshold)

    # Tags for prefix out
    if fullKmers:
        tag_kf = "F"
    elif lowVarThreshold:
        tag_kf = "V"
    else:
        tag_kf = "S"

    # OutDir folder
    outdir = os.path.join(outdir,metagenome)
    makedirs(outdir, mode=0o700, exist_ok=True)
    outdir = os.path.join(outdir,tag_kf)

# Part 1 - K-mers profile extraction
################################################################################

    if host != "none":
        # Reference Database and Host
        k_profile_database, k_profile_host = build_load_save_data((database_seq_file, database_cls_file),
            (host_seq_file, host_cls_file),
            outdir,
            database,
            k = k_length,
            full_kmers = fullKmers,
            low_var_threshold = lowVarThreshold
        )
    else:
        # Reference Database Only
        k_profile_database = build_load_save_data((database_seq_file, database_cls_file),
            host,
            outdir,
            database,
            k = k_length,
            full_kmers = fullKmers,
            low_var_threshold = lowVarThreshold
        )

    # Metagenome to analyse
    k_profile_metagenome = build_load_save_data(metagenome_seq_file,
        "none",
        outdir,
        metagenome,
        k = k_length,
        full_kmers = fullKmers,
        low_var_threshold = lowVarThreshold
    )

# Part 2 - Binary classification of bacteria / prokaryote sequences
################################################################################

# TESTER OTHER CLASSIFIERS
    for bacteria_classifier in ["linearsvm","attention","lstm","cnn","deeplstm"]:
        print("Testing classifier {}".format(bacteria_classifier))
        if host == "none":
            bacterial_metagenome = bacteria_extraction(k_profile_metagenome,
                k_profile_database,
                k_length,
                outdir,
                database,
                classifier = bacteria_classifier,
                batch_size = training_batch_size,
                verbose = verbose,
                cv = bacteria_cv,
                saving_host = bacteria_saving_host,
                saving_unclassified = bacteria_saving_unclassified
                )
        else:
            bacterial_metagenome = bacteria_extraction(k_profile_metagenome,
                (k_profile_database, k_profile_host),
                k_length,
                outdir,
                database,
                classifier = bacteria_classifier,
                batch_size = training_batch_size,
                verbose = verbose,
                cv = bacteria_cv,
                saving_host = bacteria_saving_host,
                saving_unclassified = bacteria_saving_unclassified
                )

# Part 3 - Multiclass classification of bacterial sequences
################################################################################

# MAYBE ADD PARAMETERS FOR CLASSIFIERS

# MAYBE ADD STEP FOR ABUNDANCE

# Part 4 - Classification refinement / flexible classification
################################################################################

# convert identification en np.array/list/dict de nb reads pr chaque sp

# Part 5 - (OPTIONAL) New sequences identification / clustering
################################################################################
