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
from ray.air.config import ScalingConfig
from ray.air.integrations.keras import ReportCheckpointCallback
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from ray.train.tensorflow import TensorflowTrainer, TensorflowCheckpoint

# Tuning
from ray.air.config import RunConfig

# Predicting
from tensorflow.keras.models import load_model
from ray.train.tensorflow import TensorflowPredictor
from ray.train.batch_predictor import BatchPredictor

# Data
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

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

    # Data preprocessing
    #########################################################################################################

    def preprocess(self, ds, scaling = False, scaler_file = None):
        print('preprocess')
        # Labels encoding
        self._encoder = ModelLabelEncoder(self.taxa)
        self._encoder.fit(ds)

        # Labels mapping
        labels = list(self._encoder.stats_[f'unique_values({self.taxa})'].keys())
        self._nb_classes = len(labels)
        self._encoded = np.arange(len(labels))
        labels = np.append(labels, 'Unknown')
        self._encoded = np.append(self._encoded, -1)

        for (label, encoded) in zip(labels, self._encoded):
            self._labels_map[label] = encoded
        
        # Class weights
        self._weights = self._compute_weights()

    # Models training
    #########################################################################################################

    def fit(self, datasets):
        print('fit')
        # Preprocessing loop
        for name, ds in datasets.items():
            # ds = ds.drop_columns(['id'])
            ds = self._encoder.transform(ds)
            # ds = self._scaler.transform(ds)
            ds = ds.materialize()
            datasets[name] = ds

        if self._nb_GPU > 0:
            self._fit_GPU(datasets)
        else:
            self._fit_CPU(datasets)

    def _fit_CPU(self, datasets):
        # Training parameters
        train_params = {
            'batch_size': self.batch_size,
            'epochs': self._training_epochs,
            'size': self._nb_kmers,
            'nb_cls': self._nb_classes,
            'model': self.classifier,
            'weights': self._weights
        }

        # Define trainer / tuner
        self._trainer = TensorflowTrainer(
            train_loop_per_worker=train_func_CPU,
            train_loop_config=train_params,
            scaling_config=ScalingConfig(
                trainer_resources={'CPU': self._nb_CPU_data},
                num_workers=self._n_workers,
                use_gpu=self._use_gpu,
                resources_per_worker={
                    'CPU': self._nb_CPU_per_worker
                }
            ),
            run_config=RunConfig(
                name=self.classifier,
                local_dir=self._workdir,
            ),
            datasets=datasets,
        )

        training_result = self._trainer.fit()
        self._model_ckpt = training_result.best_checkpoints[0][0]

    def _fit_GPU(self, datasets):
        # Training parameters
        train_params = {
            'batch_size': self.batch_size,
            'epochs': self._training_epochs,
            'size': self._nb_kmers,
            'nb_cls': self._nb_classes,
            'taxa': self.taxa,
            'workdir':self._workdir,
            'model': self.classifier,
            'weights': self._weights
        }

        self._model_ckpt = train_func_GPU(datasets, train_params)

    # Models predicting
    #########################################################################################################

    def predict(self, ds):
        print('predict')
        # Predict with model
        probabilities = self._predict_proba(ds)
        # Convert predictions to labels
        predictions = self._get_abs_pred(probabilities)
        # Return decoded labels
        return self._label_decode(predictions)
    
    def predict_proba(self, ds, threshold = 0.8):
        print('predict_proba')
        # Predict with model
        probabilities = self._predict_proba(ds)
        # Convert predictions to labels with threshold
        predictions = self._get_threshold_pred(probabilities, threshold)
        # Return decoded labels
        return self._label_decode(predictions)

    def _predict_proba(self, ds):
        print('_predict_proba')
        if ds.count() > 0:
            if len(ds.schema().names) > 1:
                col_2_drop = [col for col in ds.schema().names if col != TENSOR_COLUMN_NAME]
                ds = ds.drop_columns(col_2_drop)

            ds = ds.materialize()

            if self._nb_GPU > 0:
                probabilities = self._predict_proba_GPU(ds)
            else:
                probabilities = self._predict_proba_CPU(ds)
            
            return probabilities
        else:
            raise ValueError('No data to predict')

    def _predict_proba_CPU(self, ds):
        print('_predict_proba_CPU')
        self._predictor = BatchPredictor.from_checkpoint(
            self._model_ckpt,
            TensorflowPredictor,
            model_definition = lambda: build_model(self.classifier, self._nb_classes, self._nb_kmers)
        )
        predictions = self._predictor.predict(
            data = ds,
            feature_columns = [TENSOR_COLUMN_NAME],
            batch_size = self.batch_size,
        )

        probabilities = _unwrap_ndarray_object_type_if_needed(predictions.to_pandas()['predictions'])

        return probabilities
    
    def _predict_proba_GPU(self, ds):
        print('_predict_proba_GPU')
        model = load_model(self._model_ckpt)
        probabilities = []
        for batch in ds.iter_tf_batches(batch_size = self.batch_size):
            probabilities.extend(model.predict(batch[TENSOR_COLUMN_NAME]))

        return probabilities

    @abstractmethod
    def _get_abs_pred(self):
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

    # Data
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
        # Training
        history = model.fit(
            x = batch_train,
            validation_data = batch_val,
            callbacks = [ReportCheckpointCallback()],
            class_weight = weights,
            verbose = 0
        )
        # Checkpointing
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

def train_func_GPU(datasets, config):
    # Parameters
    batch_size = config.get('batch_size', 128)
    epochs = config.get('epochs', 10)
    size = config.get('size')
    nb_cls = config.get('nb_cls')
    taxa = config.get('taxa')
    workdir = config.get('workdir')
    model = config.get('model')
    weights = config.get('weights')

    checkpoint = os.path.join(workdir, model)

    # Data
    train_ds = datasets['train']
    val_ds = datasets['validation']

    # Convert datasets to tensorflow ds & generator
    train_ds = train_ds.iterator().to_tf(
        feature_columns = TENSOR_COLUMN_NAME,
        label_columns = LABELS_COLUMN_NAME,
        batch_size = batch_size
    )
    val_ds = val_ds.to_tf(
        feature_columns = TENSOR_COLUMN_NAME,
        label_columns = LABELS_COLUMN_NAME,
        batch_size = batch_size
    )

    # Model construction
    model = build_model(model, nb_cls, size)

    # Callbacks
    model_file = os.path.join(checkpoint, taxa, '{epoch:03d}.hdf5')
    model_csv = os.path.join(checkpoint, taxa, 'training_log.csv')
    modelckpt = ModelCheckpoint(filepath=model_file, monitor='val_loss', save_best_only=True, mode='auto')
    early = EarlyStopping(monitor='val_loss', mode='auto', patience=10)
    csv = CSVLogger(model_csv)

    # Training
    hist = model.fit(
        train_ds,
        epochs = epochs,
        validation_data = val_ds,
        callbacks = [modelckpt, early, csv],
        class_weight = weights,
        verbose = 0
    )

    # Checkpointing
    best_model = np.argmin(hist.history['val_loss']) + 1
    best_model = f'{best_model:03d}.hdf5'
    best_model = os.path.join(checkpoint, taxa, best_model)
    
    return best_model


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

