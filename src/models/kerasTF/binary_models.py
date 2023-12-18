import os
import gc
import warnings
import numpy as np
import pandas as pd

# Preprocessing
from ray.data.preprocessors import LabelEncoder, Chain
from models.encoders.model_label_encoder import ModelLabelEncoder
from models.preprocessors.min_max_scaler import TensorMinMaxScaler
from models.encoders.one_hot_tensor_encoder import OneHotTensorEncoder
from models.preprocessors.tfidf_transformer import TensorTfIdfTransformer

# Parent class / models
from models.kerasTF.models import KerasTFModels
from models.kerasTF.build_neural_networks import *

# Training
import tensorflow as tf
from ray.air import session
# from ray.air.integrations.keras import Callback
from ray.air.config import ScalingConfig
from models.kerasTF.models import train_func, build_model
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

class KerasTFBinaryModels(KerasTFModels):
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
        self._nb_classes = 2
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

        if self.classifier == 'attention':
            print('Training bacterial / host classifier based on Attention Weighted Neural Network')
        elif self.classifier == 'lstm':
            print('Training bacterial / host classifier based on Shallow LSTM Neural Network')
        elif self.classifier == 'deeplstm':
            print('Training bacterial / host classifier based on Deep LSTM Neural Network')
        
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
        
        # Scaling
        # self._scaler = TensorTfIdfTransformer(
        #     features = self.kmers,
        #     file = scaler_file
        # )
        # self._scaler = TensorMinMaxScaler(self._nb_kmers)
        # self._scaler.fit(ds)
    
    # Model training
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
            train_loop_per_worker=train_func,
            train_loop_config=train_params,
            scaling_config=ScalingConfig(
                trainer_resources={'CPU': self._nb_CPU_data},
                num_workers=self._n_workers,
                use_gpu=self._use_gpu,
                resources_per_worker={
                    'CPU': self._nb_CPU_per_worker,
                    'GPU': self._nb_GPU_per_worker
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

    # Model predicting
    #########################################################################################################

    def predict(self, ds):
        print('predict')
        # Predict with model
        predictions = self._predict_proba(ds)
        # Convert predictions to labels
        predictions = self._get_abs_pred(predictions)
        # Return decoded labels
        return self._label_decode(predictions)

    def predict_proba(self, ds, threshold = 0.8):
        print('predict_proba')
        # Predict with model
        predictions = self._predict_proba(ds)
        # Convert predictions to labels with threshold
        predictions = self._get_threshold_pred(predictions, threshold)
        # Return decoded labels
        return self._label_decode(predictions)
    
    def _predict_proba(self, ds):
        if ds.count() > 0:
            if len(ds.schema().names) > 1:
                col_2_drop = [col for col in ds.schema().names if col != TENSOR_COLUMN_NAME]
                ds = ds.drop_columns(col_2_drop)

            # Preprocess
            # ds = self._scaler.transform(ds)
            ds = ds.materialize()

            self._predictor = BatchPredictor.from_checkpoint(
                self._model_ckpt,
                TensorflowPredictor,
                model_definition = lambda: build_model(self.classifier, self._nb_classes, self._nb_kmers)
            )
            if self._nb_GPU > 0:
                predictions = self._predictor.predict(
                    data = ds,
                    feature_columns = [TENSOR_COLUMN_NAME],
                    batch_size = self.batch_size,
                    num_gpus_per_worker = self._nb_GPU_per_worker
                )
            else:
                predictions = self._predictor.predict(
                    data = ds,
                    feature_columns = [TENSOR_COLUMN_NAME],
                    batch_size = self.batch_size,
                    num_cpus_per_worker = self._nb_CPU_per_worker
                )
            return predictions
        else:
            raise ValueError('No data to predict')
        
    def _get_abs_pred(self, predictions):
        print('_get_abs_pred')
        def map_predicted_label(ds):
            ds = np.ravel(ds['predictions'])
            threshold = 0.5
            predict = pd.DataFrame({
                'proba': ds,
                'predicted_label': np.full(len(ds), -1)
            })
            predict.loc[predict['proba'] > threshold, 'predicted_label'] = 1
            predict.loc[predict['proba'] < threshold, 'predicted_label'] = 0
            return {'predictions' : predict['predicted_label'].to_numpy(dtype = np.int32)}
                
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
            ds = np.ravel(ds['predictions'])
            lower_threshold = 0.5 - (threshold * 0.5)
            upper_threshold = 0.5 + (threshold * 0.5)
            predict = pd.DataFrame({
                'proba': ds,
                'predicted_label': np.full(len(ds), -1)
            })
            predict.loc[predict['proba'] >= upper_threshold, 'predicted_label'] = 1
            predict.loc[predict['proba'] <= lower_threshold, 'predicted_label'] = 0
            return {'predictions' : predict['predicted_label'].to_numpy(dtype = np.int32)}
        
        predict = []
        predictions = predictions.map_batches(
            lambda batch : map_predicted_label(batch, threshold),
            batch_format = 'numpy',
            batch_size = self.batch_size
        )
        for row in predictions.iter_rows():
            predict.append(row['predictions'])

        return predict