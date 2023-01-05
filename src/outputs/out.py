import os
import gzip

import numpy as np
from Bio import SeqIO
import pandas as pd
from subprocess import run

__author__ = 'Nicolas de Montigny'

__all__ = ['Outputs']

class Outputs():
    """
    ----------
    Attributes
    ----------

    database_kmers : dictionnary
        The database of K-mers and their associated classes used for classifying the dataset

    results_dir : string
        Path to a folder to output results

    k : int
        Length of k-mers used for classification
    
    classifier : string
        Name of the classifier used

    dataset : string
        Name of the dataset

    host : string
        Name of the host if there is one

    classified_data : dictionnary
        The classes that were predicted by models

    ----------
    Methods
    ----------

    mpa_style : Generates an abundances table in tsv format similar to metaphlan's output
        No parameters required

    kronagram : Generates a Kronagram (interactive tree) in html format
        No parameters required

    report : Generates a full report on identification of classified sequences
        No parameters required

    """
    def __init__(
        self,
        database_kmers,
        results_dir,
        k,
        classifier,
        dataset,
        host,
        classified_data
    ):
        # Third-party path
        self._krona_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'KronaTools','scripts','ImportText.pl')
        self._perl_loc = run('which perl', shell = True, capture_output = True, text = True).stdout.strip('\n')
        # Variables
        self.host = host
        self.dataset = dataset
        self.classified_data = classified_data
        self.taxas = database_kmers['taxas']
        self.order = classified_data['sequence']
        self.data_labels = pd.DataFrame(
            database_kmers['classes'],
            columns = database_kmers['taxas']
        )
        # File names
        self._summary_file = '{}summary_K{}_{}_{}.csv'.format(results_dir, k, classifier, dataset)
        self._krona_file = '{}kronagram_K{}_{}_{}.csv'.format(results_dir, k, classifier, dataset)
        self._krona_out = '{}kronagram_K{}_{}_{}.html'.format(results_dir, k, classifier, dataset)
        self._report_file = '{}report_K{}_{}_{}.csv'.format(results_dir, k, classifier, dataset)
        self._mpa_file = '{}mpa_K{}_{}_{}.csv'.format(results_dir, k, classifier, dataset)
        # Initialize empty
        self._abundances = {}
        # Get abundances used for other outputs
        self._get_abundances()
        # Auto output summary
        self._summary_table()


    def _get_abundances(self):
        for taxa in self.order:
            df = self.classified_data[taxa]['classification']
            self._abundances[taxa] = {
                'counts': df.value_counts(subset = [taxa]),
                'total': df.value_counts(subset = [taxa]).sum()
            }     

    # Summary file of operations / counts & proportions of reads at each steps
    def _summary_table(self):
        # Raw abundances
        reads_total = (len(self.classified_data['domain']['classified_ids']) + len(self.classified_data['domain']['unknown_ids']))
        reads_bacteria = len(self.classified_data['domain']['classified_ids'])
        if self.host is not None:
            reads_host = len(self.classified_data['domain']['host_ids'])
            reads_classified = (reads_bacteria + reads_host)
        else:
            reads_classified = reads_bacteria
        reads_unknown = len(self.classified_data['domain']['unknown_ids'])
        rows = [
            'Total number of reads',
            'Total number of classified reads',
            'Total Number of unknown reads'
        ]
        values_raw = [
            reads_total,
            reads_classified,
            reads_unknown
        ]
        # Relative abundances
        values_rel = [
            np.NaN,
            ((reads_classified/reads_total)*100),
            ((reads_unknown/reads_total)*100)
        ]
        for taxa in self.order:
            if taxa == 'domain':
                rows.append('bacteria')
                values_raw.append(reads_bacteria)
                values_rel.append((reads_bacteria/reads_total)*100)
                if self.host is not None:
                    rows.append(self.host)
                    values_raw.append(reads_host)
                    values_rel.append((reads_host/reads_total)*100)
            else:
                rows.append(taxa)
                values_raw.append(self._abundances[taxa]['total'])
                values_rel.append((self._abundances[taxa]['total']/reads_total)*100)

        df = pd.DataFrame({
            'Taxa' : rows,
            'Abundance': values_raw,
            'Relative abundance (%)': values_rel
        })

        df.to_csv(self._summary_file, na_rep = '', header = True)
        print(f'Summary table saved to {self._summary_file}')

    # Kronagram / interactive tree
    def kronagram(self):
        self._create_krona_file()
        cmd = '{} {} {} -o {} -n {}'.format(self._perl_loc, self._krona_path, self._krona_file, self._krona_out, self.dataset)
        run(cmd, shell = True)
        print(f'Kronagram saved to {self._krona_out}')

    def _create_krona_file(self):
        # Reverse order of columns
        db_labels = self.data_labels.copy(deep = True)
        db_labels = db_labels[db_labels.columns[::-1]]
        taxas = self.order.copy()
        if 'domain' in taxas:
            taxas.remove('domain')
            taxas.append('domain')
        
        df = pd.DataFrame(columns=[taxa for taxa in reversed(taxas)])
        
        for taxa in taxas:
            abund_per_tax = pd.DataFrame(self._abundances[taxa]['counts'])
            abund_per_tax.reset_index(level = 0, inplace = True)
            abund_per_tax.columns = [taxa, 'abundance']
            abund_per_tax = abund_per_tax.join(db_labels, how = 'left', on = taxa)
            abund_per_tax.index = abund_per_tax['abundance'] # Index is abundance
            df = pd.concat([df, abund_per_tax], axis = 0, ignore_index = False) # Keep abundance on index when concatenating
        
        #taxas.insert(0, 'abundance')
        #df.reset_index(level = 0, inplace = True)
        df.to_csv(self._krona_file, na_rep = '', header = False, index = True)
        print(f'Abundance file required for Kronagram saved to {self._krona_file}')

    # Report file of classification of each id
    def report(self):
        lst_ids = []
        db_labels = self.data_labels.copy(deep = True)
        taxas = self.order.copy()
        if 'domain' in taxas:
            taxas.remove('domain')
            taxas.append('domain')
        taxas.append('id')

        taxas = [taxa for taxa in reversed(taxas)]
        
        df = pd.DataFrame(columns = taxas)
        for taxa in taxas:
            tmp_df = self.classified_data[taxa]['classification']
            lst_ids.extend(self.classified_data[taxa]['classified_ids'])
            for classif in tmp_df[taxa]:
                row = db_labels[db_labels[taxa] == classif]
                df = pd.concat([df, row], axis = 0, ignore_index = True)
            db_labels = db_labels.drop(taxa, axis = 1)
            db_labels = db_labels.drop_duplicates()
        
        df['id'] = lst_ids
        df.to_csv(self._report_file, na_rep = '', header = True, index = False)
        print(f'Classification report saved to {self._report_file}')

# TODO : Convertir cette fonction pour mpa-style
    # Bacteria abundance tables / relative abundance vs total bacteria
    def mpa_style(self):
        

        lst_taxa = list(self._abundances.keys())
        lst_taxa.insert(0, 'unknown')
        lst_nb_reads = np.zeros(len(lst_taxa) + 1, dtype = object)
        lst_rel_abund = np.zeros(len(lst_taxa) + 1, dtype = object)
        reads_total = (len(self.classified_data['domain']['classified_ids']) + len(self.classified_data['domain']['unknown_ids']))
        lst_nb_reads[0] = len(self.classified_data['domain']['unknown_ids'])

        if 'domain' in lst_taxa:
            lst_taxa.remove('domain')
        if 'host' in self._abundances:
            lst_taxa.remove('host')
        
        if 'domain' in self.classified_data.keys():
            nb_total_bacteria = len(self.classified_data['domain']['classified_ids'])
        else:
            nb_total_bacteria = 0
            for taxa in lst_taxa:
                nb_total_bacteria += self._abundances[taxa]['total']

        print(lst_taxa)
        for taxa in lst_taxa:
            print(taxa)
            lst_taxa[lst_taxa.index(taxa)] = []
            lst_taxa[lst_taxa.index(taxa)].append(taxa)
            lst_taxa[lst_taxa.index(taxa)].extend(list(self._abundances[taxa]['counts'].index))
            lst_nb_reads[lst_taxa.index(taxa)] = [self._abundances[taxa]['total']]
            lst_nb_reads[lst_taxa.index(taxa)].extend(list(self._abundances[taxa]['counts'].values))
            
        lst_taxa = np.ravel(lst_taxa)
        lst_nb_reads = np.ravel(lst_nb_reads)
        lst_rel_abund = (lst_nb_reads / reads_total) * 100

        df = pd.DataFrame({
            'Taxonomic classification': lst_taxa,
            'Number of reads': lst_nb_reads,
            'Relative Abundance (%)': lst_rel_abund
        })

        df.to_csv(self._abund_file, na_rep = '', header = True, index = False)
        print(f'mpa-style file saved to {self._mpa_file}')

