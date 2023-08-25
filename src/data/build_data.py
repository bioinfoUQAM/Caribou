
import os

from utils import load_Xy_data, save_Xy_data
from data.kmers_collection import KmersCollection

__author__ = 'Nicolas de Montigny'

__all__ = ['build_load_save_data', 'build_Xy_data', 'build_X_data']


def build_load_save_data(file, hostfile, prefix, dataset, host, kmers_list = None, k = 20):
    # Test for which dataset to build k-mers and return it
    # Database + Host
    if isinstance(file, tuple) and isinstance(hostfile, tuple) and kmers_list is None:
        db_data = build_kmers_db(file, dataset, prefix, k)
        host_data = build_kmers_db(hostfile, host, prefix, k, db_data['kmers'])
        return db_data, host_data
    # Database only
    elif isinstance(file, tuple) and kmers_list is None:
        return build_kmers_db(file, dataset, prefix, k)
    # Host only
    elif isinstance(hostfile, tuple) and kmers_list is not None:
        return build_kmers_db(hostfile, host, prefix, k, kmers_list)
    # Dataset only
    elif not isinstance(file, tuple) and kmers_list is not None:
        return build_kmers_dataset(file, dataset, prefix, k, kmers_list)
    else:
        raise ValueError('Invalid parameters combinaison for k-mers profile building')

def build_kmers_db(file, dataset, prefix, k, kmers_list = None):
    print(f'{dataset} {k}-mers profile')
    # Generate the names of files
    Xy_file = os.path.join(prefix, f'Xy_genome_{dataset}_data_K{k}')
    data_file = os.path.join(prefix, f'Xy_genome_{dataset}_data_K{k}.npz')
    # Load db file if already exists
    if os.path.isfile(data_file):
        data = load_Xy_data(data_file)
    else:
        # Build kmers collections with known classes and taxas
        collection = KmersCollection(
            file[0],
            Xy_file,
            k,
            file[1],
            kmers_list,
        )
        collection.compute_kmers()

        data = {
                # Data in a dictionnary
                'profile': collection.Xy_file,  # Kmers profile
                'ids': collection.ids,  # Ids of profiles
                'classes': collection.classes,  # Class labels
                'kmers': collection.kmers_list,  # Features
                'taxas': collection.taxas,  # Known taxas for classification
                'fasta': collection.fasta,  # Fasta file -> simulate reads if cv
        }
        save_Xy_data(data, data_file)
    return data
       
def build_kmers_dataset(file, dataset, prefix, k, kmers_list):
    print(f'{dataset} {k}-mers profile')
    # Generate the names of files
    Xy_file = os.path.join(prefix, f'Xy_genome_{dataset}_data_K{k}')
    data_file = os.path.join(prefix, f'Xy_genome_{dataset}_data_K{k}.npz')
    # Load dataset file if already exists
    if os.path.isfile(data_file):
        data = load_Xy_data(data_file)
    else:
        # Build kmers collection with unknown classes
        collection = KmersCollection(
            file,
            Xy_file,
            k,
            cls_file = None,
            kmers_list = kmers_list
        )
        collection.compute_kmers()
        # Data in a dictionnary
        data = {
            'profile' : collection.Xy_file,
            'ids' : collection.ids,
            'kmers' : collection.kmers_list
        }
        save_Xy_data(data, data_file)
    return data