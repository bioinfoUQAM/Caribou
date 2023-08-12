
import os

from utils import load_Xy_data, save_Xy_data
from data.kmers_collection import KmersCollection

__author__ = 'Nicolas de Montigny'

__all__ = ['build_load_save_data', 'build_Xy_data', 'build_X_data']


def build_load_save_data(file, hostfile, prefix, dataset, host, kmers_list=None, k=20):
    # Declare data variables as none
    data = None
    data_host = None
    # Generate the names of files
    Xy_file = os.path.join(prefix, 'Xy_genome_{}_data_K{}'.format(dataset, k))
    data_file = os.path.join(prefix, 'Xy_genome_{}_data_K{}.npz'.format(dataset, k))
    Xy_file_host = os.path.join(prefix, 'Xy_genome_{}_data_K{}'.format(host, k))
    data_file_host = os.path.join(prefix, 'Xy_genome_{}_data_K{}.npz'.format(host, k))

    # Load file if already exists
    if os.path.isfile(data_file) and os.path.isfile(data_file_host) and isinstance(hostfile, tuple):
        data = load_Xy_data(data_file)
        data_host = load_Xy_data(data_file_host)
    elif os.path.isfile(data_file) :
        data = load_Xy_data(data_file)
    else:
        # Build Xy_data of database
        print('Database Xy_data, k = {}'.format(k))
        data = build_Xy_data(
            fasta = file[0],
            csv = file[1],
            k = k,
            Xy_file = Xy_file,
            kmers_list=None,
        )
        save_Xy_data(data, data_file)

        # Assign kmers_list to variable after extracting database data
        if kmers_list is None:
            kmers_list = data['kmers']

        # Build Xy_data of host
        if isinstance(hostfile, tuple) and kmers_list is not None:
            # Build Xy_data to drive
            print('Host/simulated Xy_data, k = {}'.format(k))
            data_host = build_Xy_data(
                fasta = file[0],
                csv = file[1],
                k = k,
                Xy_file = Xy_file_host,
                kmers_list = kmers_list,

            )
            save_Xy_data(data_host, data_file_host)

        # Build X_data of dataset to analyse
        if not isinstance(file, tuple) and not isinstance(hostfile, tuple) and kmers_list is not None:
            print('Dataset X_data, k = {}'.format(k))
            data = build_X_data(
                #  seq_data,
                fasta = file,
                k = k,
                Xy_file = Xy_file,
                kmers_list = kmers_list
            )
            save_Xy_data(data, data_file)

    if data is not None and data_host is None:
        return data
    elif data is None and data_host is not None:
        return data_host
    else:
        return data, data_host

# Build kmers collections with known classes and taxas
def build_Xy_data(fasta, csv, k, Xy_file, kmers_list = None):
    data = {}

    collection = KmersCollection(
        fasta,
        Xy_file,
        k,
        csv,
        kmers_list,
    )
    collection.compute_kmers()

    # Data in a dictionnary
    data['profile'] = collection.Xy_file  # Kmers profile
    data['ids'] = collection.ids  # Ids of profiles
    data['classes'] = collection.classes  # Class labels
    data['kmers'] = collection.kmers_list  # Features
    data['taxas'] = collection.taxas  # Known taxas for classification
    data['fasta'] = collection.fasta  # Fasta file -> simulate reads if cv

    return data

# Build kmers collection with unknown classes
def build_X_data(fasta, k, X_file, kmers_list):
    data = {}

    collection = KmersCollection(
        fasta,
        X_file,
        k,
        kmers_list
    )
    collection.compute_kmers()

    # Data in a dictionnary
    data['profile'] = collection.Xy_file
    data['ids'] = collection.ids
    data['kmers'] = collection.kmers_list

    return data
