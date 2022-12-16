import os
import ray
import warnings
import numpy as np
import modin.pandas as pd

from glob import glob
from shutil import rmtree

# Preprocessing
from models.ray_tensor_min_max import TensorMinMaxScaler
from ray.data.preprocessors import BatchMapper, Concatenator, LabelEncoder, Chain, OneHotEncoder

# Parent class / models
from models.ray_utils import ModelsUtils
from models.kerasTF.build_neural_networks import *

# Training
import tensorflow as tf
from ray.air import session, Checkpoint
from ray.air.callbacks.keras import Callback
from ray.air.config import ScalingConfig, CheckpointConfig
from ray.train.tensorflow import TensorflowTrainer, TensorflowCheckpoint, prepare_dataset_shard

# Tuning
from ray.air.config import RunConfig
from ray.tune import SyncConfig

# Predicting
from ray.train.tensorflow import TensorflowPredictor
from ray.train.batch_predictor import BatchPredictor

__author__ = 'Nicolas de Montigny'

__all__ = ['KerasTFModel']

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
            k,
            taxa,
            kmers_list,
            verbose
        )
        # Parameters
        # Initialize hidden
        self._training_epochs = training_epochs
        # Initialize empty
        self._nb_classes = None
        # Computing variables
        if len(tf.config.list_physical_devices('GPU')) > 0:
            self._use_gpu = True
            self._n_workers = len(tf.config.list_physical_devices('GPU'))
        else:
            self._use_gpu = False
        self._n_workers = int(np.floor(os.cpu_count()*.8))
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

    def _training_preprocess(self, X, y):
        print('_training_preprocess')
        self._preprocessor = TensorMinMaxScaler(self.kmers)
        self._preprocessor.fit(X)
        df = X.to_modin()
        df[taxa] = y[taxa]
        df = ray.data.from_modin(df)
        self._label_encode(df, y)
        return df

    def _label_encode(self, df, y):
        self._nb_classes = len(np.unique(y.to_modin()[self.taxa]))
        self._label_encode_define(df)

        encoded = []
        encoded.append(-1)
        labels = ['unknown']
        for k, v in self._encoder.preprocessors[0].stats_['unique_values({})'.format(self.taxa)].items():
            encoded.append(v)
            labels.append(k)

        self._labels_map = zip(labels, encoded)

    def _label_encode_define(self, df):
        print('_label_encode_multiclass')
        self._encoder = Chain(
            LabelEncoder(self.taxa),
            OneHotEncoder([self.taxa]),
            Concatenator(
                output_column_name='labels',
                include=['{}_{}'.format(self.taxa, i) for i in range(self._nb_classes)]
            )
        )
        self._encoder.fit(df)

    def _label_decode(self, predict):
        print('_label_decode')
        decoded = pd.Series(np.empty(len(predict), dtype=object))
        for label, encoded in self._labels_map:
            decoded[predict == encoded] = label

        return decoded

    def train(self, X, y, kmers_ds, cv = True):
        print('train')
        df = self._training_preprocess(X, y)
        if cv:
            self._cross_validation(df, kmers_ds)
        else:
            df_train, df_val = df.train_test_split(0.2, shuffle=True)
            df_val = self._sim_4_cv(df_val, kmers_ds, 'validation')
            df_train = df_train.drop_columns(['id'])
            df_val = df_val.drop_columns(['id'])
            datasets = {'train': ray.put(df_train), 'validation': ray.put(df_val)}
            self._fit_model(datasets)

    def _cross_validation(self, df, kmers_ds):
        print('_cross_validation')

        df_train, df_test = df.train_test_split(0.2, shuffle = True)
        df_train, df_val = df_train.train_test_split(0.1, shuffle = True)

        df_val = self._sim_4_cv(df_val, kmers_ds, '{}_val'.format(self.dataset))
        df_test = self._sim_4_cv(df_test, kmers_ds, '{}_test'.format(self.dataset))

        df_train = df_train.drop_columns(['id'])
        df_test = df_test.drop_columns(['id'])
        df_val = df_val.drop_columns(['id'])

        datasets = {'train' : ray.put(df_train), 'validation' : ray.put(df_val)}
        self._fit_model(datasets)

        df_test = self._encoder.preprocessors[0].transform(df_test)
        y_true = df_test.to_modin()[self.taxa]
        y_pred = self.predict(df_test.drop_columns([self.taxa]), cv = True)

        for file in glob(os.path.join(os.path.dirname(kmers_ds['profile']), '*sim*')):
            if os.path.isdir(file):
                rmtree(file)
            else:
                os.remove(file)

        self._cv_score(y_true, y_pred)

    def _fit_model(self, datasets):
        print('_fit_model')
        for name, ds in datasets.items():
            ds = ray.get(ds)
            ds = self._preprocessor.transform(ds)
            ds = self._encoder.transform(ds)
            datasets[name] = ds

        # Training parameters
        self._train_params = {
            'batch_size': self.batch_size,
            'epochs': self._training_epochs,
            'size': self._nb_kmers,
            'nb_cls':self._nb_classes,
            'model': self.classifier
        }
        # Define trainer / tuner
        self._trainer = TensorflowTrainer(
            train_loop_per_worker = train_func,
            train_loop_config = self._train_params,
            scaling_config = ScalingConfig(
                num_workers = self._n_workers,
                use_gpu = self._use_gpu
            ),
            run_config = RunConfig(
                name = self.classifier,
                local_dir = self._workdir,
                sync_config = SyncConfig(syncer=None),
                checkpoint_config = CheckpointConfig(
                    checkpoint_score_attribute = 'loss',
                    checkpoint_score_order = 'min'
                )
            ),
            datasets = datasets
        )
        # Train / tune execution
        training_result = self._trainer.fit()
        self._model_ckpt = training_result.best_checkpoints[0][0]

    def predict(self, df, threshold = 0.8, cv = False):
        print('predict')
        if df.count() > 0:
            df = self._preprocessor.transform(df)
            # Define predictor
            self._predictor = BatchPredictor.from_checkpoint(
                self._model_ckpt,
                TensorflowPredictor,
                model_definition = lambda : build_model(self.classifier, self._nb_classes, len(self.kmers))
            )
            # Make predictions
            predictions = self._predictor.predict(
                df,
                feature_columns = ['__value__'],
                batch_size = self.batch_size
            )
            predictions = self._prob_2_cls(predictions, threshold)

            if cv:
                return predictions
            else:
                return self._label_decode(predictions)
        else:
            raise ValueError('No data to predict')

    def _prob_2_cls(self, predictions, threshold):
        print('_prob_2_cls')
        def map_predicted_label_binary(df):
            lower_threshold = 0.5 - (threshold * 0.5)
            upper_threshold = 0.5 + (threshold * 0.5)
            predict = pd.DataFrame({
                'proba': df['predictions'],
                'predicted_label': np.full(len(df), -1)
            })
            predict.loc[predict['proba'] >= upper_threshold, 'predicted_label'] = 1
            predict.loc[predict['proba'] <= lower_threshold, 'predicted_label'] = 0
            return pd.DataFrame(predict['predicted_label'])

        def map_predicted_label_multiclass(df):
            predict = pd.DataFrame({
                'best_proba': [df['predictions'][i][np.argmax(df['predictions'][i])] for i in range(len(df))],
                'predicted_label': [np.argmax(df['predictions'][i]) for i in range(len(df))]
            })
            predict.loc[predict['best_proba'] < threshold, 'predicted_label'] = -1
            return pd.DataFrame(predict['predicted_label'])
       
        if self._nb_classes == 2:
            mapper = BatchMapper(map_predicted_label_binary, batch_format = 'pandas')
        else:
            mapper = BatchMapper(map_predicted_label_multiclass, batch_format = 'pandas')
        predict = mapper.transform(predictions)
        predict = np.ravel(np.array(predict.to_modin()))
        
        return predict

# Training/building function outside of the class as mentioned on the Ray discussion
# https://discuss.ray.io/t/statuscode-resource-exhausted/4379/16
################################################################################

def train_func(config):
    batch_size = config.get('batch_size', 128)
    epochs = config.get('epochs', 10)
    size = config.get('size')
    nb_cls = config.get('nb_cls')
    model = config.get('model')

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = build_model(model, nb_cls, size)

    train_data = session.get_dataset_shard('train')
    val_data = session.get_dataset_shard('validation')

    def to_tf_dataset(data, batch_size):
        def to_tensor_iterator():
            for batch in data.iter_tf_batches(
                batch_size=batch_size, dtypes=tf.float32,
            ):
                yield batch['__value__'], batch['labels']

        output_signature = (
            tf.TensorSpec(shape=(None, size), dtype=tf.float32),
            tf.TensorSpec(shape=(None, nb_cls), dtype=tf.int64),
        )
        tf_data = tf.data.Dataset.from_generator(
            to_tensor_iterator, output_signature=output_signature
        )
        return prepare_dataset_shard(tf_data)

    results = []
    for epoch in range(epochs):
        tf_train_data = to_tf_dataset(
            data = train_data,
            batch_size = batch_size
        )
        tf_val_data = to_tf_dataset(
            data = val_data,
            batch_size = batch_size
        )
        history = model.fit(
            tf_train_data,
            validation_data = tf_val_data,
            callbacks=[Callback()],
            verbose=0
        )
        results.append(history.history)
        session.report({
            'accuracy' : history.history['accuracy'][0],
            'loss' : history.history['loss'][0],
            'val_accuracy' : history.history['val_accuracy'][0],
            'val_loss' : history.history['val_loss'][0]
            },
            checkpoint = TensorflowCheckpoint.from_model(model)
        )

def build_model(classifier, nb_cls, nb_kmers):
    if classifier == 'attention':
        clf = build_attention(nb_kmers)
    elif classifier == 'lstm':
        clf = build_LSTM(nb_kmers)
    elif classifier == 'deeplstm':
        clf = build_deepLSTM(nb_kmers)
    elif classifier == 'lstm_attention':
        clf = build_LSTM_attention(nb_kmers, nb_cls)
    elif classifier == 'cnn':
        clf = build_CNN(nb_kmers, nb_cls)
    elif classifier == 'widecnn':
        clf = build_wideCNN(nb_kmers, nb_cls)
    return clf