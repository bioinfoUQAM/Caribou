#!/usr/bin/R
library(reticulate)

.onLoad <- function(libname, pkgname) {
  caribou = import("Caribou", delay_load = TRUE)
}

caribou = function(config_file):
  caribou$Caribou$caribou(r_to_py(c("Caribou.py",config_file), convert = T))

build_data = function(db_file, host_file, prefix, dataset, kmers_list=None, k=4){
  return(py_to_r(caribou$data$build_data$build_load_save_data(r_to_py(db_file),
                                                              r_to_py(host_file),
                                                              r_to_py(prefix),
                                                              r_to_py(dataset),
                                                              r_to_py(kmers_list),
                                                              r_to_py(k))))
}

bacteria_extraction = function(metagenome_k_mers, database_k_mers, k, outdirs, dataset, classifier = "attention", batch_size = 32, verbose = 1, cv = 1, saving_host = 1, saving_unclassified = 1, n_jobs = 1){
  return(py_to_r(caribou$models$bacteria_extraction$bacteria_extraction(r_to_py(metagenome_k_mers),
                                                                        r_to_py(database_k_mers),
                                                                        r_to_py(k),
                                                                        r_to_py(outdirs),
                                                                        r_to_py(dataset),
                                                                        r_to_py(classifier),
                                                                        r_to_py(batch_size),
                                                                        r_to_py(verbose),
                                                                        r_to_py(cv),
                                                                        r_to_py(saving_host),
                                                                        r_to_py(saving_unclassified),
                                                                        r_to_py(n_jobs))))
}

classification = function(classified_data, database_k_mers, k, outdirs, dataset, classifier = "lstm_attention", batch_size = 32, threshold = 0.8, verbose = 1, cv = 1, n_jobs = 1){
  return(py_to_r(caribou$models$classification$bacteria_classification(r_to_py(classified_data),
                                                                       r_to_py(database_k_mers),
                                                                       r_to_py(k),
                                                                       r_to_py(outdirs),
                                                                       r_to_py(dataset),
                                                                       r_to_py(classifier),
                                                                       r_to_py(batch_size),
                                                                       r_to_py(threshold),
                                                                       r_to_py(verbose),
                                                                       r_to_py(cv),
                                                                       r_to_py(n_jobs))))
}

outputs = function(database_kmers, results_dir, k, classifier = "lstm_attention", dataset, host, classified_data, seq_file, abundance_stats = True, kronagram = True, full_report = True){
  caribou$outputs$outputs$outputs(r_to_py(database_kmers),
                                  r_to_py(results_dir),
                                  r_to_py(k),
                                  r_to_py(classifier),
                                  r_to_py(dataset),
                                  r_to_py(host),
                                  r_to_py(classified_data),
                                  r_to_py(seq_file),
                                  r_to_py(abundance_stats),
                                  r_to_py(kronagram),
                                  r_to_py(full_report))
}

simulation = function(fasta, classes, genomes, reads, type, prefix){
  opt = py_dict(keys = c('fasta', 'classes', 'genomes', 'reads', 'type', 'prefix'),
                values = c(fasta, classes, genomes, reads, type, prefix),
                convert = TRUE)
  caribou$supplement$simulation$simulation(opt)
}
