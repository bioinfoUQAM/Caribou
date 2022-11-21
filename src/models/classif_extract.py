
import os
import sys
import ray
import pickle
import cloudpickle

import pandas as pd

from models.ray_sklearn import SklearnModel
from models.ray_keras_tf import KerasTFModel
from utils import load_Xy_data, save_Xy_data

from models.classif_utils import ClassificationUtils

__author__ = 'Nicolas de Montigny'

__all__ = ['ClassificationExtraction']

class ClassificationExtraction(ClassificationUtils):
    """
    Class for extracting bacterial sequences from metagenomes using a trained model

    ----------
    Attributes
    ----------

    ----------
    Methods
    ----------

    """
    def __init__(
        self,
        database_k_mers,
        k,
        outdirs,
        dataset,
        training_epochs = 100,
        classifier = 'deeplstm',
        batch_size = 32,
        verbose = True,
        cv = True
    ):
        super().__init__(
            self,
            None,
            database_k_mers,
            k,
            outdirs,
            dataset,
            training_epochs,
            classifier,
            batch_size,
            verbose,
            cv
        )
        # Parameters
        self.classify_data = metagenome_k_mers
        self.taxas = ['bacteria']
        # Empty initializations
        self.host = False
        self.model = None
        

        def train_model():
            print('todo')

        def classify(self):
            print('todo')

        def extract_bacteria(self):
                if not os.path.isfile(model_file):
                    # Get training dataset and assign to variables
                    # Keep only classes of sequences that were not removed in kmers extraction
                    if classifier == 'onesvm' and isinstance(database_k_mers, tuple):
                        print('Classifier One Class SVM cannot be used with host data!\nEither remove host data from config file or choose another bacteria extraction method.')
                        sys.exit()
                    elif classifier == 'onesvm' and not isinstance(database_k_mers, tuple):
                        model = SklearnModel(classifier, dataset, outdirs['models_dir'], outdirs['results_dir'], batch_size, k, 'domain', database_k_mers['kmers'], verbose)
                        X_train = ray.data.read_parquet(database_k_mers['profile']).window(blocks_per_window = 10)
                        X_train = unpack_kmers(X_train, database_k_mers['kmers'])
                        y_train = pd.DataFrame(
                            {'domain': pd.DataFrame(database_k_mers['classes'], columns=database_k_mers['taxas']).loc[:, 'domain'].astype('string').str.lower(),
                            'id': database_k_mers['ids']}
                        )
                    elif classifier != 'onesvm' and isinstance(database_k_mers, tuple):
                        database_k_mers = merge_database_host(database_k_mers[0], database_k_mers[1])
                        if classifier in ['attention','lstm','deeplstm']:
                            model = KerasTFModel(classifier, dataset, outdirs['models_dir'], outdirs['results_dir'], batch_size, training_epochs, k, 'domain', database_k_mers['kmers'], verbose)
                        elif classifier == 'linearsvm':
                            model = SklearnModel(classifier, dataset, outdirs['models_dir'], outdirs['results_dir'], batch_size, k, 'domain', database_k_mers['kmers'], verbose)
                        else:
                            print('Bacteria extractor unknown !!!\n\tModels implemented at this moment are :\n\tBacteria isolator :  One Class SVM (onesvm)\n\tBacteria/host classifiers : Linear SVM (linearsvm)\n\tNeural networks : Attention (attention), Shallow LSTM (lstm) and Deep LSTM (deeplstm)')
                            sys.exit()
                        X_train = ray.data.read_parquet(database_k_mers['profile']).window(blocks_per_window = 10)
                        X_train = unpack_kmers(X_train, database_k_mers['kmers'])
                        y_train = pd.DataFrame(
                            {'domain': pd.DataFrame(database_k_mers['classes'], columns=database_k_mers['taxas']).loc[:, 'domain'].astype('string').str.lower(),
                            'id': database_k_mers['ids']}
                        )
                    else:
                        print('Only classifier One Class SVM can be used without host data!\nEither add host data in config file or choose classifier One Class SVM.')
                        sys.exit()

                    model.train(X_train, y_train, database_k_mers, cv)
                    with open(model_file, 'wb') as handle:
                        pickle.dump(cloudpickle.dumps(model), handle)
                else:
                    with open(model_file, 'rb') as handle:
                        model = pickle.load(cloudpickle.loads(handle))

                # Classify sequences into bacteria / unclassified / host and build k-mers profiles for bacteria
                if metagenome_k_mers is not None:
                    classified_data['bacteria'] = extract(metagenome_k_mers['profile'], model, verbose)
                    save_Xy_data(classified_data['bacteria'], bacteria_data_file)

            return classified_data
