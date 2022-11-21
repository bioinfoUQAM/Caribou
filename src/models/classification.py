
def bacteria_classification(classified_data, database_k_mers, k, outdirs, dataset, training_epochs = 100, classifier = 'lstm_attention', batch_size = 32, threshold = 0.8, verbose = True, cv = True, classifying = False):


                # Classify sequences into taxa and build k-mers profiles for classified and unclassified data
                # Keep previous taxa to reclassify only unclassified reads at a higher taxonomic level
                if classifying is True:
                    if previous_taxa_unclassified is None:
                        if verbose:
                            print('Classifying bacteria sequences at {} level'.format(taxa))
                        df = ray.data.read_parquet(classified_data['bacteria']['profile'])
                        classified_data[taxa], previous_taxa_unclassified = classify(df, model, taxa, classified_kmers_file, unclassified_kmers_file, threshold = threshold, verbose = verbose)
                    else:
                        if verbose:
                            print('Classifying bacteria sequences at {} level'.format(taxa))
                        classified_data[taxa], previous_taxa_unclassified = classify(previous_taxa_unclassified['profile'], model, threshold, verbose)

                    save_Xy_data(classified_data[taxa], classified_kmers_file)
                    save_Xy_data(previous_taxa_unclassified, unclassified_kmers_file)
                    classified_data['order'].append(taxa)

    return classified_data

def classify(df_file, model, threshold = 0.8, verbose = True):
# TO DO: CONSERVATION DES CLASSIFICATIONS DANS DES DOSSIERS RAY DS EN FICHIERS PARQUET POUR CHAQUE NIVEAU TAXONOMIQUE
# TO DO: CRÉATION + APPEND DE RAY DS POUR LES CLASSES IDENTIFIÉES PAR NIVEAU TAXONOMIQUE -> FACILITE OUTPUTS
    if verbose:
        print('Extracting predicted sequences at {} taxonomic level'.format(taxa))

    df = ray.data.read_parquet(df_file).window(blocks_per_window=10)

    classified_data = {}

    pred = model.predict(df, threshold)

    df = df.to_pandas()

    # Make sure classes are writen in lowercase
    pred['class'] = pred['class'].str.lower()

    df_classified = ray.data.from_pandas(df[pred['class'].str.notequals('unknown')])
    df_unclassified = ray.data.from_pandas(df[pred['class'].str.match('unknown')])

    # Save / add to classified/unclassified data
    try:
        df_classified.to_parquet(classified_kmers_file)
        classified_data['classified'] = {}
        classified_data['classified']['profile'] = str(classified_kmers_file)
    except:
        if verbose:
            print('No classified data at {} taxonomic level, cannot save it to file or add it to classified data'.format(taxa))
    try:
        df_unclassified.to_parquet(unclassified_kmers_file)
        classified_data['unclassified'] = {}
        classified_data['unclassified']['profile'] = str(unclassified_kmers_file)
    except:
        if verbose:
            print('No unclassified data at {} taxonomic level, cannot save it to file or add it to unclassified data'.format(taxa))

    return classified_data, unclassified_data
