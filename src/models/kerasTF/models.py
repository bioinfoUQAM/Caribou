import os
import gc
import warnings
import numpy as np
import pandas as pd

# Class construction
from abc import ABC, abstractmethod

# Preprocessing
from ray.data.preprocessors import LabelEncoder, Chain
from models.encoders.model_label_encoder import ModelLabelEncoder
from models.encoders.one_hot_tensor_encoder import OneHotTensorEncoder
from models.preprocessors.tfidf_transformer import TensorTfIdfTransformer

# Parent class / models
from models.models_utils import ModelsUtils
from models.kerasTF.build_neural_networks import *

# Training
import tensorflow as tf
from ray.air import session
# from ray.air.integrations.keras import Callback
from ray.air.config import ScalingConfig
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

class KerasTFModels(ModelsUtils, ABC):
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
        self._nb_CPU_data = int(os.cpu_count() * 0.2) # 6
        self._nb_CPU_training = int(os.cpu_count() - self._nb_CPU_data) # 26
        self._nb_GPU = len(tf.config.list_physical_devices('GPU')) # 6
        # Initialize empty
        self._nb_CPU_per_worker = 0
        self._nb_GPU_per_worker = 0
        # Computing variables
        if self._nb_GPU > 0:
            self._use_gpu = True
            self._n_workers = self._nb_GPU #6
            self._nb_CPU_per_worker = int(self._nb_CPU_training / self._n_workers) # 4
            self._nb_GPU_per_worker = 1
        else:
            self._use_gpu = False
            self._n_workers = int(self._nb_CPU_training * 0.2)
            self._nb_CPU_per_worker = int(int(self._nb_CPU_training * 0.8) / self._n_workers)

    @abstractmethod
    def preprocess(self):
        """
        """
        
    @abstractmethod
    def fit(self, datasets):
        """
        """

    @abstractmethod
    def predict(self, ds):
        """
        """

    @abstractmethod
    def predict_proba(self):
        """
        """

    @abstractmethod
    def _get_threshold_pred(self):
        """
        """

# Training/building function outside of the class as mentioned on the Ray discussion
# https://discuss.ray.io/t/statuscode-resource-exhausted/4379/16
################################################################################

# Data streaming in PipelineDataset for larger than memory data, should prevent OOM
# https://docs.ray.io/en/latest/ray-air/check-ingest.html#enabling-streaming-ingest
# Smaller nb of workers + bigger nb CPU_per_worker + smaller batch_size to avoid memory overload
# https://discuss.ray.io/t/ray-sgd-distributed-tensorflow/261/8

def train_func_CPU(config):
    # Parameters
    batch_size = config.get('batch_size', 128)
    epochs = config.get('epochs', 10)
    size = config.get('size')
    nb_cls = config.get('nb_cls')
    model = config.get('model')
    weights = config.get('weights')

    # Model construction
    model = build_model(model, nb_cls, size)

    train_data = session.get_dataset_shard('train')
    val_data = session.get_dataset_shard('validation')

    for _ in range(epochs):
        batch_train = train_data.to_tf(
            feature_columns = TENSOR_COLUMN_NAME,
            label_columns = LABELS_COLUMN_NAME,
            batch_size = batch_size,
            local_shuffle_buffer_size = batch_size,
            local_shuffle_seed = int(np.random.randint(1,10000, size = 1))
        )
        batch_val = val_data.to_tf(
            feature_columns = TENSOR_COLUMN_NAME,
            label_columns = LABELS_COLUMN_NAME,
            batch_size = batch_size,
            local_shuffle_buffer_size = batch_size,
            local_shuffle_seed = int(np.random.randint(1,10000, size = 1))
        )
        history = model.fit(
            x = batch_train,
            validation_data = batch_val,
            callbacks = [ReportCheckpointCallback()],
            class_weight = weights,
            verbose = 0
        )
        session.report({
            'accuracy': history.history['accuracy'][0],
            'loss': history.history['loss'][0],
            'val_accuracy': history.history['val_accuracy'][0],
            'val_loss': history.history['val_loss'][0],
        },
            checkpoint=TensorflowCheckpoint.from_model(model)
        )
        gc.collect()
        tf.keras.backend.clear_session()
    del model
    gc.collect()
    tf.keras.backend.clear_session()

def train_func_GPU(config):
    # Parameters
    batch_size = config.get('batch_size', 128)
    epochs = config.get('epochs', 10)
    size = config.get('size')
    nb_cls = config.get('nb_cls')
    model = config.get('model')
    weights = config.get('weights')

    # Model construction
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_model(model, nb_cls, size)

    train_data = session.get_dataset_shard('train')
    val_data = session.get_dataset_shard('validation')

    for _ in range(epochs):
        batch_train = train_data.to_tf(
            feature_columns = TENSOR_COLUMN_NAME,
            label_columns = LABELS_COLUMN_NAME,
            batch_size = batch_size,
            local_shuffle_buffer_size = batch_size,
            local_shuffle_seed = int(np.random.randint(1,10000, size = 1))
        )
        batch_val = val_data.to_tf(
            feature_columns = TENSOR_COLUMN_NAME,
            label_columns = LABELS_COLUMN_NAME,
            batch_size = batch_size,
            local_shuffle_buffer_size = batch_size,
            local_shuffle_seed = int(np.random.randint(1,10000, size = 1))
        )
        history = model.fit(
            x = batch_train,
            validation_data = batch_val,
            callbacks = [ReportCheckpointCallback()],
            class_weight = weights,
            verbose = 0
        )
        session.report({
            'accuracy': history.history['accuracy'][0],
            'loss': history.history['loss'][0],
            'val_accuracy': history.history['val_accuracy'][0],
            'val_loss': history.history['val_loss'][0],
        },
            checkpoint=TensorflowCheckpoint.from_model(model)
        )
        gc.collect()
        tf.keras.backend.clear_session()
    del model
    gc.collect()
    tf.keras.backend.clear_session()

def build_model(classifier, nb_cls, nb_kmers):
    if classifier == 'attention':
        model = build_attention(nb_kmers)
    elif classifier == 'lstm':
        model = build_LSTM(nb_kmers)
    elif classifier == 'deeplstm':
        model = build_deepLSTM(nb_kmers)
    elif classifier == 'lstm_attention':
        model = build_LSTM_attention(nb_kmers, nb_cls)
    elif classifier == 'cnn':
        model = build_CNN(nb_kmers, nb_cls)
    elif classifier == 'widecnn':
        model = build_wideCNN(nb_kmers, nb_cls)
    return model

