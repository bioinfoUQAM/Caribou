# /usr/bin/R
library(reticulate)

.onLoad <- function(libname, pkgname) {
  caribou = import("Caribou", delay_load = TRUE)
}

build_data = function(file, hostfile, prefix, dataset, kmers_list=None, k=4, full_kmers=False, low_var_threshold=None){
  caribou$data$build_data$build_load_save_data(file, hostfile, prefix, dataset, kmers_list, k, full_kmers, low_var_threshold)
}

bacteria_extraction = function(metagenome_k_mers, database_k_mers, k, outdirs, dataset, classifier = "deeplstm", batch_size = 32, verbose = 1, cv = 1, saving_host = 1, saving_unclassified = 1, n_jobs = 1){
  caribou$models$bacteria_extraction$bacteria_extraction(metagenome_k_mers, database_k_mers, k, outdirs, dataset, classifier, batch_size, verbose, cv, saving_host, saving_unclassified, n_jobs)
}

classification = function(classified_data, database_k_mers, k, outdirs, dataset, classifier = "lstm_attention", batch_size = 32, threshold = 0.8, verbose = 1, cv = 1, n_jobs = 1){
  caribou$models$classification$bacteria_classification(classified_data, database_k_mers, k, outdirs, dataset, classifier, batch_size, threshold, verbose, cv, n_jobs)
}

outputs = function(database_kmers, results_dir, k, classifier, dataset, host, classified_data, seq_file, abundance_stats = True, kronagram = True, full_report = True){
  caribou$outputs$outputs$outputs(database_kmers, results_dir, k, classifier, dataset, host, classified_data, seq_file, abundance_stats, kronagram, full_report)
}
