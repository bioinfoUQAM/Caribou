
from models.classif_utils import ClassificationUtils

__author__ = 'Nicolas de Montigny'

__all__ = ['ClassificationTaxonomic']

class ClassificationTaxonomic(ClassificationUtils):
     """
    Class for taxonomic classification of bacterial sequences from metagenomes using a trained model

    ----------
    Attributes
    ----------

    ----------
    Methods
    ----------

    """
    def __init__(
        self,
        classified_data,
        database_k_mers,
        k,
        outdirs,
        dataset,
        training_epochs = 100,
        classifier = 'lstm_attention',
        batch_size = 32,
        threshold = 0.8,
        verbose = True,
        cv = True,
        classifying = False
    ):
        super().__init__(
            self,
            classified_data,
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
        self.threshold = threshold
        self.classifying = classifying
        self.taxas = database_k_mers['taxas'].copy()
        self.taxas.remove('domain')
        
        # Empty initializations
        self.classify_data = None

        def train_model():
            print('todo')
        
        def classify(self):
            print('todo')