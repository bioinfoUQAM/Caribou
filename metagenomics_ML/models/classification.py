
__author__ = "nicolas"

def bacterial_classification(bacteria_k_mers, database_k_mers, k, prefix, dataset, classifier = "svm", verbose = 1):
    classified_file = prefix + "_K{}_{}_Xy_classified_database_{}_data.hdf5.bz2".format(k, classifier, dataset) # Pandas df en output?
    clf_file = prefix + "_K{}_{}_bacteria_identification_classifier_{}_model.joblib".format(k, classifier, dataset)


# RECURSIVITY / LOOP FOR TAXA RANK (FLEX) CLASSIFICATION OF unclassified
# ~for taxa in list:
    clf = np.array()
    classified = {}
    unclassified = {}

# NOT SUR IF USE 1 FILE CONTAINING ALL MODELS/CLASSIFIED/UNCLASSIFIED OR SPLIT INTO TAXAS...

    # Load classified data if already exists
    if os.path.isfile(classified_file):
        classified = load_Xy_data(classified_file)

    else:
        # Get training dataset
        database_k_mers["X_train"] = pd.DataFrame(StandardScaler().fit_transform(database_k_mers["X_train"]), columns = database_k_mers["kmers_list"], index = database_k_mers["ids"])
# NEED MORE RANKS FOR Y LABELS -> SPECIFY RANK IN LOOP
        database_k_mers["y_train"] = pd.DataFrame(database_k_mers["y_train"].astype(np.int32), index = database_k_mers["ids"])

        # Training data
        X_train = database_k_mers["X_train"]
        y_train = database_k_mers["y_train"]

# ADD TAXA RANKS
        # Load classifier if already trained
        if os.path.isfile(clf_file):
            clf = load(clf_file)

        else:
            # Train/test classifier
            clf_taxa = training(X_train, y_train, classifier = classifier, verbose = verbose)

        # Classify sequences into taxa / unclassified and return k-mers profiles + classification
        classified_taxa, unclassified_taxa = classify(clf_taxa, k_mers, taxa, verbose = verbose)

    if saving_mode != "none":
        save_data_clf(classified, unclassified, clf, classified_kmers_file, unclassified_kmers_file, clf_file, saving_mode)

    return
    # Bacteria Xy data
    # Multiple classifiers
    # out == np.array de bacterial sequences + nb reads de chaque
        # Classified et non-classified -> save to file et return

def training():
    print("Training")

def classify(clf, k_mers, taxa, verbose = 1):
    if taxa == "species":
        classified, unclassified = classify_species(clf, k_mers)
    elif taxa == "genus":
        classified, unclassified = classify_genus(clf, k_mers)
    elif taxa == "family":
        classified, unclassified = classify_family(clf, k_mers)
    elif taxa == "order":
        classified, unclassified = classify_order(clf, k_mers)

    return classified, unclassified

def classify_species(clf, k_mers, verbose = 1):
    print("classify_species")

def classify_genus(clf, k_mers, verbose = 1):
    print("classify_genus")

def classify_family(clf, k_mers, verbose = 1):
    print("classify_family")

def classify_order(clf, k_mers, verbose = 1):
    print("classify_order")

# CHANGE PARAMETERS TO BE MORE PRECISE / INSTINCTIVE IN CONFIG FILE

# Save extracted k-mers profiles for classified and unclassified
# Depends on wich saving mode given by user
def save_data_clf(classified, unclassified, clf, classified_kmers_file, unclassified_kmers_file, clf_file, saving_mode = "all"):
    if saving_mode == "all":
        save_Xy_data(classified, classified_kmers_file)
        save_Xy_data(unclassified, unclassified_kmers_file)
        dump(clf, clf_file)
    elif saving_mode == "both":
        save_Xy_data(classified, classified_kmers_file)
        save_Xy_data(unclassified, unclassified_kmers_file)
    elif saving_mode == "classified_model":
        save_Xy_data(classified, classified_kmers_file)
        dump(clf, clf_file)
    elif saving_mode == "unclassified_model":
        save_Xy_data(unclassified, unclassified_kmers_file)
        dump(clf, clf_file)
    elif saving_mode == "classified":
        save_Xy_data(classified, classified_kmers_file)
    elif saving_mode == "unclassified":
        save_Xy_data(unclassified, unclassified_kmers_file)
    elif saving_mode == "model":
        dump(clf, clf_file)
