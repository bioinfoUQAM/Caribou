import os
import atexit
import warnings
import numpy as np

# Preprocessing
from ray.data.preprocessors import MinMaxScaler, LabelEncoder, Chain, SimpleImputer, OneHotEncoder, Concatenator

# Parent class / models
from models.ray_utils import ModelsUtils
from models.build_neural_networks import *

# Training
import tensorflow as tf
from ray.air import session, Checkpoint
from ray.air.callbacks.keras import Callback
from ray.air.config import ScalingConfig, CheckpointConfig
from ray.train.tensorflow import TensorflowTrainer, prepare_dataset_shard


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
    # https://docs.ray.io/en/master/ray-air/examples/tfx_tabular_train_to_serve.html
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

    def __init__(self, classifier, dataset, outdir_model, outdir_results, batch_size, training_epochs, k, taxa, kmers_list, verbose):
        super().__init__(classifier, dataset, outdir_results, batch_size, k, taxa, kmers_list, verbose)
        # Parameters
        self.outdir_model = outdir_model
        if classifier in ['attention','lstm','deeplstm']:
            self.clf_file = '{}bacteria_binary_classifier_K{}_{}_{}_model'.format(outdir_model, k, classifier, dataset)
        else:
            self.clf_file = '{}{}_multiclass_classifier_K{}_{}_{}_model'.format(outdir_model, taxa, k, classifier, dataset)
        # Initialize hidden
        self._training_epochs = training_epochs
        # Initialize empty
        self._nb_classes = None
        self._use_gpu = False
        # Variables for training with Ray
        self._strategy = tf.distribute.MultiWorkerMirroredStrategy()
        # if len(tf.config.list_physical_devices('GPU')) > 0:
        #     self._use_gpu = True
        #     self._n_workers = len(tf.config.list_physical_devices('GPU'))
        # else:
        #     self._use_gpu = False

    def _training_preprocess(self, X, y):
        print('_training_preprocess')
        df = X.add_column([self.taxa, 'id'], lambda x: y)
        self._preprocessor = Chain(
            SimpleImputer(
                self.kmers,
                strategy='constant',
                fill_value=0
                ),
            MinMaxScaler(self.kmers),
            Concatenator(
                output_column_name='features',
                include=self.kmers
            )
        )
        self._label_encode(df, y)

        return df

    def _label_encode(self, df, y):
        if self.classifier in ['attention', 'lstm', 'deeplstm']:
            self._label_encode_binary(df)
        elif self.classifier in ['lstm_attention', 'cnn', 'widecnn']:
            self._label_encode_multiclass(df)

        encoded = []
        encoded.append(-1)
        labels = ['unknown']
        for k, v in self._encoder.preprocessors[0].stats_['unique_values(domain)'].items():
            encoded.append(v)
            labels.append(k)

        self._labels_map = zip(labels, encoded)
        if self.classifier in ['attention', 'lstm', 'deeplstm']:
            self._nb_classes = 1
        else:
            self._nb_classes = len(np.unique(y[self.taxa]))

    def _label_encode_binary(self, df):
        print('_label_encode_binary')
        self._encoder = Chain(
            LabelEncoder(self.taxa),
            Concatenator(
                output_column_name='labels',
                include=[self.taxa]
            )
        )
        self._encoder.fit(df)

    def _label_encode_multiclass(self, df):
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

    def _cross_validation(self, df, kmers_ds):
        print('_cross_validation')

        df_train, df_test = df.train_test_split(0.2, shuffle = True)
        df_train, df_val = df_train.train_test_split(0.1, shuffle = True)

        df_val = self._sim_4_cv(df_val, kmers_ds, '{}_val'.format(self.dataset))
        df_test = self._sim_4_cv(df_test, kmers_ds, '{}_test'.format(self.dataset))

        df_train = df_train.drop_columns(['id'])

        df_train = self._encoder.transform(df_train)
        df_val = self._encoder.transform(df_val)
        df_test = self._encoder.transform(df_test)

        datasets = {'train' : df_train, 'validation' : df_val}
        self._fit_model(datasets)

        y_true = df_test.to_pandas()[self.taxa]
        y_pred = self.predict(df_test.drop_columns(['id',self.taxa]), cv = True)

        rmtree(sim_data['profile'])
        for file in glob(os.path.join(sim_outdir, '*sim*')):
            os.remove(file)

        self._cv_score(y_true, y_pred)


    def _build(self, classifier, nb_cls, nb_kmers):
        print('_build')
        with self._strategy.scope():
            atexit.register(strategy._extended._cross_device_ops._pool.close) # type: ignore
            atexit.register(strategy._extended._host_cross_device_ops._pool.close) #type: ignore
            if classifier == 'attention':
                print('Training bacterial / host classifier based on Attention Weighted Neural Network')
                clf = build_attention(nb_kmers)
            elif classifier == 'lstm':
                print('Training bacterial / host classifier based on Shallow LSTM Neural Network')
                clf = build_LSTM(nb_kmers)
            elif classifier == 'deeplstm':
                print('Training bacterial / host classifier based on Deep LSTM Neural Network')
                clf = build_deepLSTM(nb_kmers)
            elif classifier == 'lstm_attention':
                print('Training multiclass classifier based on Deep Neural Network hybrid between LSTM and Attention')
                clf = build_LSTM_attention(nb_kmers, nb_cls)
            elif classifier == 'cnn':
                print('Training multiclass classifier based on CNN Neural Network')
                clf = build_CNN(nb_kmers, nb_cls)
            elif classifier == 'widecnn':
                print('Training multiclass classifier based on Wide CNN Network')
                clf = build_wideCNN(nb_kmers, nb_cls)
        return clf

    def _fit_model(self, datasets):
        print('_fit_model')
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
            train_loop_per_worker=self._train_func,
            train_loop_config=self._train_params,
            scaling_config=ScalingConfig(num_workers=self._n_workers, use_gpu=self._use_gpu),
            run_config=RunConfig(
                name=self.classifier,
                local_dir=self.outdir_model,
                sync_config=SyncConfig(syncer=None),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute='accuracy',
                    checkpoint_score_order='max'
                )
            ),
            datasets=datasets
        )
        # Train / tune execution
        training_result = self._trainer.fit()
        self._model_ckpt = training_result.best_checkpoint[0][0]

    def _train_func(self, config):
        print('_train_func')
        batch_size = config.get('batch_size', 128)
        epochs = config.get('epochs', 10)
        size = config.get('size')
        nb_cls = config.get('nb_cls')
        model = config.get('model')

        model = self._build(model, nb_cls, size)

        data = session.get_dataset_shard('train')

        def to_tf_dataset(data, batch_size):
            def to_tensor_iterator():
                for batch in data.iter_tf_batches(
                    batch_size=batch_size, dtypes=tf.float32,
                ):
                    yield batch['features'], batch['labels']

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
            tf_data = to_tf_dataset(data, batch_size)
            history = model.fit(tf_data, verbose=0, callbacks=[Callback()])
            results.append(history.history)
            session.report(
                dict(accuracy=history.history['accuracy'][0], loss=history.history['loss'][0]),
                checkpoint=Checkpoint.from_dict(
                    dict(model=model, epoch=epoch, model_weights=model.get_weights())
                )
            )

        return results

    def predict(self, df, threshold = 0.8, cv = False):
        print('predict')
        if not cv:
            df = self._predict_preprocess(df)
        # Define predictor
        self._predictor = BatchPredictor.from_checkpoint(
            self._model_ckpt,
            TensorflowPredictor,
            model_definiton = self._clf
        )
        # Make predictions
        predictions = self._predictor.predict(
            df,
            feature_columns = ['features'],
            batch_size = self.batch_size
        )
        if cv:
            return predictions
        else:
            return self._label_decode(predictions, threshold)

    # Overcharge to serialize class
    def __reduce__(self):
        deserializer = self.__class__
        serialized_data = (self.classifier, self.dataset, self.outdir_model, self.outdir_results, self.batch_size, self._training_epochs, self.k, self.taxa, self.kmers, self.verbose)

        return deserializer, serialized_data
