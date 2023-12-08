import os
import gc
import warnings
import numpy as np
import pandas as pd

# Preprocessing
from ray.data.preprocessors import LabelEncoder, Chain
from models.encoders.model_label_encoder import ModelLabelEncoder
from models.encoders.one_hot_tensor_encoder import OneHotTensorEncoder
from models.preprocessors.tfidf_transformer import TensorTfIdfTransformer

# Parent class / models
from models.kerasTF.models import KerasTFModels
from models.kerasTF.build_neural_networks import *
from models.multiclass_utils import MulticlassUtils

# Training
import tensorflow as tf
from ray.air import session
# from ray.air.integrations.keras import Callback
from ray.air.config import ScalingConfig
from models.kerasTF.models import train_func
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.train.tensorflow import TensorflowTrainer, TensorflowCheckpoint

# Tuning
from ray.air.config import RunConfig

# Predicting
from ray.train.tensorflow import TensorflowPredictor
from ray.train.batch_predictor import BatchPredictor

__author__ = 'Nicolas de Montigny'

__all__ = ['KerasTFModel']

TENSOR_COLUMN_NAME = '__value__'
LABELS_COLUMN_NAME = 'labels'

# Ignore warnings to have a more comprehensible output on stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

class KerasTFMulticlassModels(KerasTFModels, MulticlassUtils):
    """
    Class used to build, train and predict models using Ray with Keras Tensorflow backend

    ----------
    Attributes
    ----------

    clf_file : string
        Path to a file containing the trained model for this object

    nb_classes : int
        Number of classes for learning

    ----------
    Methods
    ----------

    preprocess : preprocess the data before training and splitting the original dataset in case of cross-validation

    train : train a model using the given datasets

    predict : predict the classes of a dataset
        ds : ray.data.Dataset
            Dataset containing K-mers profiles of sequences to be classified

        threshold : float
            Minimum percentage of probability to effectively classify.
            Sequences will be classified as 'unknown' if the probability is under this threshold.
            Defaults to 80%
    """

    def __init__(
        self,
        classifier,
        outdir_model,
        batch_size,
        training_epochs,
        taxa,
        kmers_list,
        csv
    ):
        super().__init__(
            classifier,
            outdir_model,
            batch_size,
            training_epochs,
            taxa,
            kmers_list,
            csv
        )
        # Parameters
        # Initialize hidden
        self._nb_CPU_data = int(os.cpu_count() * 0.2)
        self._nb_CPU_training = int(os.cpu_count() - self._nb_CPU_data)
        self._nb_GPU = len(tf.config.list_physical_devices('GPU'))
        # Initialize empty
        self._nb_classes = None
        self._nb_CPU_per_worker = 0
        self._nb_GPU_per_worker = 0
        # Computing variables
        if self._nb_GPU > 0:
            self._use_gpu = True
            self._n_workers = self._nb_GPU
            self._nb_CPU_per_worker = int(self._nb_CPU_training / self._n_workers)
            self._nb_GPU_per_worker = 1
        else:
            self._use_gpu = False
            self._n_workers = int(self._nb_CPU_training * 0.2)
            self._nb_CPU_per_worker = int(int(self._nb_CPU_training * 0.8) / self._n_workers)

    # Data preprocessing
    #########################################################################################################

    def preprocess(self, ds, scaling = False, scaler_file = None):
        print('preprocess')
        # Labels encoding
        self._encoder = Chain(
            ModelLabelEncoder(self.taxa),
            OneHotTensorEncoder(LABELS_COLUMN_NAME)
        )
        self._encoder.fit(ds)

        # Labels mapping
        labels = list(self._encoder.preprocessors[0].stats_[f'unique_values({self.taxa})'].keys())
        self._nb_classes = len(labels)
        self._encoded = np.arange(len(labels))
        labels = np.append(labels, 'Unknown')
        self._encoded = np.append(self._encoded, -1)

        for (label, encoded) in zip(labels, self._encoded):
            self._labels_map[label] = encoded
        
        # Class weights
        self._weights = self._compute_weights()
        
        # Scaling
        if scaling:
            self._scaler = TensorTfIdfTransformer(self.kmers, scaler_file)
            self._scaler.fit(ds)

    # Models training
    #########################################################################################################

    def fit(self, datasets):
        print('fit')
        # Preprocessing loop
        for name, ds in datasets.items():
            # ds = ds.drop_columns(['id'])
            ds = self._encoder.transform(ds)
            if self._scaler is not None:
                ds = self._scaler.transform(ds)
            ds = ds.materialize()
            datasets[name] = ds
        
        # One sub-model per artificial cluster of samples
        ds['train'] = self._random_split_dataset(ds['train'])

        # Checkpointing directory
        model_dir = os.path.join(self._workdir, f'{self.classifier}_{self.taxa}')
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

# TODO: train_func per model
# TODO: Confirm how it works in Jupyter Notebook
        # Distributed building & training
        if self.classifier == 'lstm_attention':
            print('Training multiclass classifier based on Deep Neural Network hybrid between LSTM and Attention')
            training_result = ds.map_groups(build_fit_lstm_attention, batch_format = 'numpy')
        elif self.classifier == 'cnn':
            print('Training multiclass classifier based on CNN Neural Network')
            training_result = ds.map_groups(build_fit_cnn, batch_format = 'numpy')
        elif self.classifier == 'widecnn':
            print('Training multiclass classifier based on Wide CNN Network')
            training_result = ds.map_groups(build_fit_widecnn, batch_format = 'numpy')

        training_result = training_result.to_pandas().to_dict('records')
        for record in training_result:
            self._model_ckpt[record['cluster']] = record['file']

    # Models predicting
    #########################################################################################################

    def predict(self, ds):
        print('predict')
        probabilities = self._predict_proba(ds)
        predictions = np.argmax(probabilities, axis = 1)
        predictions = self._label_decode(predictions)
        return predictions
    
    def predict_proba(self, ds, threshold = 0.8):
        print('predict_proba')
        probabilities = self._predict_proba(ds)
        predictions = self._get_threshold_pred(probabilities, threshold)
        return self._label_decode(predictions)

# TODO: Confirm how it works in Jupyter Notebook
    def _predict_proba(self, ds):
        print('_predict_proba')
        if ds.count() > 0:
            if self._scaler is not None:
                ds = self._scaler.transform(ds)
            # ds = ds.materialize()

            def predict_func(data):
                X = _unwrap_ndarray_object_type_if_needed(data[TENSOR_COLUMN_NAME])
                pred = np.zeros((len(X), len(self._labels_map)))
                for cluster, model_file in self._model_ckpt.items():
                    with open(model_file, 'rb') as file:
                        model = cpickle.load(file)
                    proba = model.predict_proba(X)
                    for i, cls in enumerate(model.classes_):
                        pred[:, cls] += proba[:, i]
                # pred = pred / len(self._model_ckpt)
                return {'predictions' : pred}

            probabilities = ds.map_batches(predict_func, batch_format = 'numpy')
            probabilities = _unwrap_ndarray_object_type_if_needed(probabilities.to_pandas()['predictions'])
            
            return probabilities
        else:
            raise ValueError('Empty dataset, cannot execute predictions!')

    def _get_abs_pred(self, predictions):
        print('_get_abs_pred')
        def map_predicted_label(ds):
            ds = ds['predictions']
            pred = pd.DataFrame({
                'best_proba': [np.max(arr) for arr in ds],
                'predicted_label' : [np.argmax(arr) for arr in ds]
            })

            return {'predictions' : pred['predicted_label'].to_numpy(dtype = np.int32)}
        
        predict = []
        predictions = predictions.map_batches(
            lambda batch : map_predicted_label(batch),
            batch_format = 'numpy',
            batch_size = self.batch_size
        )
        for row in predictions.iter_rows():
            predict.append(row['predictions'])

        return predict

    def _get_threshold_pred(self, predictions, threshold):
        print('_get_threshold_pred')
        def map_predicted_label(ds, threshold):
            ds = ds['predictions']
            pred = pd.DataFrame({
                'best_proba': [np.max(arr) for arr in ds],
                'predicted_label' : [np.argmax(arr) for arr in ds]
            })
            pred.loc[pred['best_proba'] < threshold, 'predicted_label'] = -1

            return {'predictions' : pred['predicted_label'].to_numpy(dtype = np.int32)}

        predict = []
        predictions = predictions.map_batches(
            lambda batch : map_predicted_label(batch, threshold),
            batch_format = 'numpy',
            batch_size = self.batch_size
        )
        for row in predictions.iter_rows():
            predict.append(row['predictions'])

        return predict

# TODO: Confirm how it works in Jupyter Notebook
def build_fit_lstm_attention(data):
    """
    LSTM-Attention NN training function
    """

def build_fit_cnn(data):
    """
    Convolution NN training function
    """

def build_fit_widecnn(data):
    """
    Wide Convolution NN training function
    """