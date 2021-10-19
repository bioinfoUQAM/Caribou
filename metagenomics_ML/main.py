#!/usr/bin/env python

from data.build_data import *
from models.bacteria_extraction import *
from models.classification import *

import pandas as pd

import sys
import configparser
import os.path
from os import makedirs

__author__ = "nicolas"

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
    k_lenght = config.getint("seq_rep", "k")
    fullKmers = config.getboolean("seq_rep", "full_kmers")
    lowVarThreshold = config.get("seq_rep", "low_var_threshold", fallback=None)

# MAYBE SOME NOT NEEDED
    # evaluation
    cv_folds = config.getint("evaluation", "cv_folds")
    eval_metric = config.get("evaluation", "eval_metric")
    avrg_metric = config.get("evaluation", "avrg_metric")

    # classifier
    bacteria_classifier = config.get("classifier", "binary_classifier")

# MAYBE SOME NOT NEEDED
    # settings
    n_mainJobs = config.getint("settings", "n_main_jobs")
    n_cvJobs = config.getint("settings", "n_cv_jobs")
    verbose = config.getint("settings", "verbose")
    training_batch_size = config.getint("settings", "training_batch_size")

    bacteria_saving = config.get("settings", "binary_save_others")
    bacteria_cv = config.get("settings", "binary_cross_val")

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

    for k_length in range(4,20):
        if host != "none":
            # Reference Database and Host
            k_profile_database = build_load_save_data((database_seq_file, database_cls_file),
                (host_seq_file, host_cls_file),
                outdir,
                database,
                k = k_lenght,
                full_kmers = fullKmers,
                low_var_threshold = lowVarThreshold
            )
        else:
            # Reference Database Only
            k_profile_database = build_load_save_data((database_seq_file, database_cls_file),
                host,
                outdir,
                database,
                k = k_lenght,
                full_kmers = fullKmers,
                low_var_threshold = lowVarThreshold
            )

        # Metagenome to analyse
        k_profile_metagenome = build_load_save_data(metagenome_seq_file,
            "none",
            outdir,
            metagenome,
            k = k_lenght,
            full_kmers = fullKmers,
            low_var_threshold = lowVarThreshold
        )

    # Part 2 - Binary classification of bacteria / prokaryote sequences
    ################################################################################

    #    for classifier in ["oneSVM","lof","multiSVM","forest","knn","lstm"]:
        bacterial_metagenome = bacteria_extraction(k_profile_metagenome,
            k_profile_database,
            k_lenght,
            outdir,
            database,
            classifier = bacteria_classifier,
            batch_size = training_batch_size,
            verbose = verbose,
            saving = bacteria_saving,
            cv = bacteria_cv
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
