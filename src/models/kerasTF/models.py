import os
import gc
import ray
import warnings
import numpy as np
import pandas as pd

from glob import glob
from shutil import rmtree

# Dimensions reduction
from models.preprocessors.tfidf_transformer import TensorTfIdfTransformer
from data.reduction.rdf_features_selection import TensorRDFFeaturesSelection
from data.reduction.truncated_svd_reduction import TensorTruncatedSVDReduction

# Preprocessing
from ray.data.preprocessors import LabelEncoder, Chain
from models.preprocessors.min_max_scaler import TensorMinMaxScaler
from models.encoders.model_label_encoder import ModelLabelEncoder
from models.encoders.one_hot_tensor_encoder import OneHotTensorEncoder

# Parent class / models
from models.models_utils import ModelsUtils
from models.kerasTF.build_neural_networks import *

# Training
import tensorflow as tf
from ray.air import session
from ray.train import DataConfig
# from ray.air.integrations.keras import Callback
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.air.config import ScalingConfig #DatasetConfig
from ray.train.tensorflow import TensorflowTrainer, TensorflowCheckpoint, prepare_dataset_shard

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

class KerasTFModel(ModelsUtils):
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
        df : ray.data.Dataset
            Dataset containing K-mers profiles of sequences to be classified

        threshold : float
            Minimum percentage of probability to effectively classify.
            Sequences will be classified as 'unknown' if the probability is under this threshold.
            Defaults to 80%

    """

    def __init__(
        self,
        classifier,
        dataset,
        outdir_model,
        outdir_results,
        batch_size,
        training_epochs,
        k,
        taxa,
        kmers_list,
        verbose
    ):
        super().__init__(
            classifier,
            dataset,
            outdir_model,
            outdir_results,
            batch_size,
            training_epochs,
            k,
            taxa,
            kmers_list,
            verbose
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

        if self.classifier == 'attention':
            print('Training bacterial / host classifier based on Attention Weighted Neural Network')
        elif self.classifier == 'lstm':
            print('Training bacterial / host classifier based on Shallow LSTM Neural Network')
        elif self.classifier == 'deeplstm':
            print('Training bacterial / host classifier based on Deep LSTM Neural Network')
        elif self.classifier == 'lstm_attention':
            print('Training multiclass classifier based on Deep Neural Network hybrid between LSTM and Attention')
        elif self.classifier == 'cnn':
            print('Training multiclass classifier based on CNN Neural Network')
        elif self.classifier == 'widecnn':
            print('Training multiclass classifier based on Wide CNN Network')

    def preprocess(self, df):
        print('preprocess')
        labels = []
        encoded = []
        for row in df.iter_rows():
            labels.append(row[self.taxa])
        self._nb_classes = len(np.unique(labels))
        if self._nb_classes == 2:
            self._encoder = ModelLabelEncoder(self.taxa)
            self._preprocessor = Chain(
                TensorTfIdfTransformer(self.kmers),
                TensorRDFFeaturesSelection(self.kmers, self.taxa),
            )
        else:
            self._encoder = Chain(
                LabelEncoder(self.taxa),
                OneHotTensorEncoder(self.taxa)
            )
            self._preprocessor = Chain(
                TensorTfIdfTransformer(self.kmers),
                TensorRDFFeaturesSelection(self.kmers, self.taxa),
            )
        
        self._encoder.fit(df)
        df = self._preprocessor.fit_transform(df)
        self._reductor = TensorTruncatedSVDReduction(self.kmers)
        self._reductor.fit(df)
        # Labels mapping
        if self._nb_classes == 2:
            labels = list(self._encoder.stats_[f'unique_values({self.taxa})'].keys())
        else:
            labels = list(self._encoder.preprocessors[0].stats_[f'unique_values({self.taxa})'].keys())
        encoded = np.arange(len(labels))
        labels = np.append(labels, 'unknown')
        encoded = np.append(encoded, -1)
        self._labels_map = zip(labels, encoded)

    def _label_decode(self, predict):
        print('_label_decode')
        decoded = pd.Series(np.empty(len(predict), dtype=object))
        for label, encoded in self._labels_map:
            decoded[predict == encoded] = label

        return np.array(decoded)

    def train(self, datasets, kmers_ds, cv = True):
        print('train')
        if cv:
            self._cross_validation(datasets, kmers_ds)
        else:
            self._fit_model(datasets)

    def _cross_validation(self, datasets, kmers_ds):
        print('_cross_validation')
        df_test = datasets.pop('test')

        self._fit_model(datasets)

        y_true = []
        for row in df_test.iter_rows():
            y_true.append(row[self.taxa])

        y_pred = self.predict(df_test.drop_columns([self.taxa]), threshold = 0.8)

        self._cv_score(y_true, y_pred)

    def _fit_model(self, datasets):
        print('_fit_model')
        # Preprocessing loop
        for name, ds in datasets.items():
            ds = ds.drop_columns(['id'])
            ds = self._encoder.transform(ds)
            ds = self._preprocessor.transform(ds)
            ds = self._reductor.transform(ds)
            datasets[name] = ds

        # Training parameters
        self._train_params = {
            'batch_size': self.batch_size,
            'epochs': self._training_epochs,
            'size': self._nb_kmers,
            'nb_cls': self._nb_classes,
            'model': self.classifier
        }

        # Define trainer / tuner
        self._trainer = TensorflowTrainer(
            train_loop_per_worker=train_func,
            train_loop_config=self._train_params,
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

    def predict(self, df, threshold=0.8):
        print('predict')
        if df.count() > 0:
            if len(df.schema().names) > 1:
                col_2_drop = [col for col in df.schema().names if col != TENSOR_COLUMN_NAME]
                df = df.drop_columns(col_2_drop)

            # Preprocess
            df = self._preprocessor.transform(df)

            self._predictor = BatchPredictor.from_checkpoint(
                self._model_ckpt,
                TensorflowPredictor,
                model_definition = lambda: build_model(self.classifier, self._nb_classes, len(self.kmers))
            )
            predictions = self._predictor.predict(
                data = df,
                batch_size = self.batch_size
            )

            # Convert predictions to labels
            predictions = self._prob_2_cls(predictions, threshold)
                
            return self._label_decode(predictions)
        else:
            raise ValueError('No data to predict')

    # Iterate over batches of predictions to transform probabilities to labels without mapping
    def _prob_2_cls(self, predictions, threshold):
        print('_prob_2_cls')
        def map_predicted_label_binary(df, threshold):
            df = np.ravel(df['predictions'])
            lower_threshold = 0.5 - (threshold * 0.5)
            upper_threshold = 0.5 + (threshold * 0.5)
            predict = pd.DataFrame({
                'proba': df,
                'predicted_label': np.full(len(df), -1)
            })
            predict.loc[predict['proba'] >= upper_threshold, 'predicted_label'] = 1
            predict.loc[predict['proba'] <= lower_threshold, 'predicted_label'] = 0
            return {'predictions' : predict['predicted_label'].to_numpy(dtype = np.int32)}
        
        def map_predicted_label_multiclass(df, threshold):
            df = df['predictions']
            pred = pd.DataFrame({
                'best_proba': [np.max(arr) for arr in df],
                'predicted_label' : [np.argmax(arr) for arr in df]
            })
            pred.loc[pred['best_proba'] < threshold, 'predicted_label'] = -1

            return {'predictions' : pred['predicted_label'].to_numpy(dtype = np.int32)}
        
        if self._nb_classes == 2:
            print('map_predicted_label_binary')
            fn = map_predicted_label_binary
        else:
            print('map_predicted_label_multiclass')
            fn = map_predicted_label_multiclass

        predict = []
        predictions = predictions.map_batches(
            lambda batch : fn(batch, threshold),
            batch_format = 'numpy',
            batch_size = self.batch_size
        )
        for row in predictions.iter_rows():
            predict.append(row['predictions'])

        return predict


# Training/building function outside of the class as mentioned on the Ray discussion
# https://discuss.ray.io/t/statuscode-resource-exhausted/4379/16
################################################################################

# Data streaming in PipelineDataset for larger than memory data, should prevent OOM
# https://docs.ray.io/en/latest/ray-air/check-ingest.html#enabling-streaming-ingest
# Smaller nb of workers + bigger nb CPU_per_worker + smaller batch_size to avoid memory overload
# https://discuss.ray.io/t/ray-sgd-distributed-tensorflow/261/8

def train_func(config):
    # Parameters
    batch_size = config.get('batch_size', 128)
    epochs = config.get('epochs', 10)
    size = config.get('size')
    nb_cls = config.get('nb_cls')
    model = config.get('model')

    

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

