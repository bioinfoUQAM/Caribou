

        # Classify sequences into bacteria / unclassified / host and build k-mers profiles for bacteria
        if metagenome_k_mers is not None:
            classified_data['bacteria'] = extract(metagenome_k_mers['profile'], model, verbose)
            save_Xy_data(classified_data['bacteria'], bacteria_data_file)

    return classified_data

def extract(df_file, model, verbose = True):
# TO DO: CONSERVATION DES CLASSIFICATIONS DANS DES DOSSIERS RAY DS EN FICHIERS PARQUET POUR CHAQUE NIVEAU TAXONOMIQUE
# TO DO: CRÉATION + APPEND DE RAY DS POUR LES CLASSES IDENTIFIÉES PAR NIVEAU TAXONOMIQUE -> FACILITE OUTPUTS
    if verbose:
        print('Extracting predicted bacteria sequences')

    df = ray.data.read_parquet(df_file).window(blocks_per_window=10)

    classified_data = {}

    pred = model.predict(df)

    df = df.to_pandas()

    # Make sure classes are writen in lowercase
    pred['class'] = pred['class'].str.lower()

    # Loop over classes to extract sequences and k-mers profiles
    for cls in ['bacteria', 'unclassified', 'host']:
        classif_kmers_file = '{}Xy_{}_database_K{}_{}_{}_data'.format(outdirs['data_dir'], cls, k, classifier, dataset)
        classif_data_file = '{}Xy_{}_database_K{}_{}_{}_data.npz'.format(outdirs['data_dir'], cls, k, classifier, dataset)
        classif_data = {}

        df_classif = ray.data.from_pandas(df[pred['class'].str.match(cls)])


    # Save / add to classified data
    try:
        df_bacteria.to_parquet(bacteria_kmers_file)
        classified_data['bacteria'] = {}
        classified_data['bacteria']['profile'] = str(bacteria_kmers_file)
    except:
        print('No bacteria data identified, cannot save it to file or add it to classified data')
    try:
        df_unclassified.to_parquet(unclassified_kmers_file)
        classified_data['unclassified'] = {}
        classified_data['unclassified']['profile'] = str(unclassified_kmers_file)
    except:
        print('No unclassified data identified, cannot save it to file or add it to unclassified data')
    try:
        df_host.to_parquet(host_kmers_file)
        classified_data['host'] = {}
        classified_data['host']['profile'] = str(host_kmers_file)
    except:
        print('No host data identified, cannot save it to file or add it to classified data')

    return classified_data
