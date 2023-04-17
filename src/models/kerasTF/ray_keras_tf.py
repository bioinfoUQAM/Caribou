import os
import gc
import ray
import warnings
import numpy as np
import pandas as pd

from glob import glob
from shutil import rmtree

# Preprocessing
from ray.data.preprocessors import LabelEncoder, Chain
from models.ray_tensor_min_max import TensorMinMaxScaler
from models.ray_tensor_max_abs import TensorMaxAbsScaler
from models.kerasTF.ray_one_hot_tensor import OneHotTensorEncoder

# Parent class / models
from models.ray_utils import ModelsUtils
from models.kerasTF.build_neural_networks import *

# Training
import tensorflow as tf
from ray.air import session
from ray.air.integrations.keras import Callback
from ray.air.config import ScalingConfig, DatasetConfig
from ray.train.tensorflow import TensorflowTrainer, TensorflowCheckpoint, prepare_dataset_shard

# Tuning
from ray.air.config import RunConfig

# Predicting
from ray.train.tensorflow import TensorflowPredictor
from ray.train.batch_predictor import BatchPredictor
from joblib import Parallel, delayed, parallel_backend

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
        for row in df.iter_rows():
            labels.append(row[self.taxa])
        self._nb_classes = len(np.unique(labels))
        self._preprocessor = Chain(
            TensorMaxAbsScaler(self.kmers),
            LabelEncoder(self.taxa),
            OneHotTensorEncoder(self.taxa),
        )
        self._preprocessor.fit(df)

    def _label_decode(self, predict):
        print('_label_decode')
        if self._labels_map is None:
            encoded = []
            encoded.append(-1)
            labels = ['unknown']
            for k, v in self._preprocessor.preprocessors[1].stats_['unique_values({})'.format(self.taxa)].items():
                encoded.append(v)
                labels.append(k)
        decoded = pd.Series(np.empty(len(predict), dtype=object))
        for label, coded in zip(labels, encoded):
            decoded[predict == coded] = label

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

        y_pred = self.predict(df_test.drop_columns([self.taxa]), threshold = 0)

        for file in glob(os.path.join(os.path.dirname(kmers_ds['profile']), '*sim*')):
            if os.path.isdir(file):
                rmtree(file)
            else:
                os.remove(file)

        self._cv_score(y_true, y_pred)

    def _fit_model(self, datasets):
        print('_fit_model')
        # Preprocessing loop
        for name, ds in datasets.items():
            print(f'dataset preprocessing : {name}')
            ds = ds.drop_columns(['id'])
            ds = self._preprocessor.transform(ds)
            datasets[name] = ds.fully_executed()

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
            dataset_config={
                'train': DatasetConfig(
                    fit=False,
                    transform=False,
                    split=True,
                    use_stream_api=True
                ),
                'validation': DatasetConfig(
                    fit=False,
                    transform=False,
                    split=True,
                    use_stream_api=False
                )
            },
            run_config=RunConfig(
                name=self.classifier,
                local_dir=self._workdir,
            ),
            datasets=datasets,
        )
        training_result = self._trainer.fit()
        self._model_ckpt = training_result.best_checkpoints[0][0]

    """
    # This is a function for using with parent class training data decomposition that may be implemented later on
    def _fit_model_multiclass(self, datasets):
        print('_fit_model')
        training_collection = datasets.pop('train')
        for name, ds in datasets.items():
            print(f'dataset preprocessing : {name}')
            ds = ds.drop_columns(['id'])
            ds = self._preprocessor.transform(ds)
            datasets[name] = ds.fully_executed()

        # Training parameters
        self._train_params = {
            'batch_size': self.batch_size,
            'epochs': self._training_epochs,
            'size': self._nb_kmers,
            'nb_cls': self._nb_classes,
            'model': self.classifier
        }

        for tax, ds in training_collection.items():
            ds = ds.drop_columns(['id'])
            ds = self._preprocessor.transform(ds)
            training_ds = {**{'train' : ds.fully_executed()}, **datasets}

            # Define trainer / tuner
            self._trainer = TensorflowTrainer(
                train_loop_per_worker = train_func,
                train_loop_config = self._train_params,
                scaling_config = ScalingConfig(
                    trainer_resources={'CPU': self._nb_CPU_data},
                    num_workers = self._n_workers,
                    use_gpu = self._use_gpu,
                    resources_per_worker={
                        'CPU': self._nb_CPU_per_worker,
                        'GPU': self._nb_GPU_per_worker
                    }
                ),
                dataset_config = {
                    'train': DatasetConfig(
                        fit = False,
                        transform = False,
                        split = True,
                        use_stream_api = True
                    ),
                    'validation': DatasetConfig(
                        fit = False,
                        transform = False,
                        split = True,
                        use_stream_api = False
                    )
                },
                run_config = RunConfig(
                    name = self.classifier,
                    local_dir = self._workdir,
                ),
                datasets = training_ds,
            )
            training_result = self._trainer.fit()
            self._models_collection[tax] = training_result.best_checkpoints[0][0]
    """
    
    def predict(self, df, threshold=0.8):
        print('predict')
        if df.count() > 0:
            if len(df.schema().names) > 1:
                col_2_drop = [col for col in df.schema().names if col != '__value__']
                df = df.drop_columns(col_2_drop)

            # Preprocess
            df = self._preprocessor.preprocessors[0].transform(df)

            print('number of classes :', self._nb_classes)

            predictor = BatchPredictor.from_checkpoint(
                self._model_ckpt,
                TensorflowPredictor,
                model_definition = lambda: build_model(self.classifier, self._nb_classes, len(self.kmers))
            )
            predictions = predictor.predict(
                data = df,
                batch_size = self.batch_size
            )

            print(predictions.to_pandas())

            # Convert predictions to labels
            predictions = self._prob_2_cls(predictions, threshold)
                
            return self._label_decode(predictions)
        else:
            raise ValueError('No data to predict')

    # Iterate over batches of predictions to transform probabilities to labels without mapping
    def _prob_2_cls(self, predictions, threshold):
        print('_prob_2_cls')
        def map_predicted_label_binary(df, threshold):
            # lower_threshold = 0.5 - (threshold * 0.5)
            # upper_threshold = 0.5 + (threshold * 0.5)
            predict = pd.DataFrame({
                'best_proba': [df['predictions'][i][np.argmax(df['predictions'][i])] for i in range(len(df))],
                'predicted_label': df["predictions"].map(lambda x: np.array(x).argmax())
            })
            # predict = pd.DataFrame({
            #     'proba': df['predictions'],
            #     'predicted_label': np.zeros(len(df), dtype = np.float32)
            # })
            print(predict)
            # predict['predicted_label'] = np.round(predict['proba'])
            # predict.loc[predict['proba'] >= upper_threshold, 'predicted_label'] = 1
            # predict.loc[predict['proba'] <= lower_threshold, 'predicted_label'] = 0
            return predict['predicted_label'].to_numpy(dtype = np.int32)
        
        def map_predicted_label_multiclass(df, threshold):
            predict = pd.DataFrame({
                'best_proba': [df['predictions'][i][np.argmax(df['predictions'][i])] for i in range(len(df))],
                'predicted_label': df["predictions"].map(lambda x: np.array(x).argmax())
            })
            predict.loc[predict['best_proba'] < threshold, 'predicted_label'] = -1
            return predict['predicted_label'].to_numpy(dtype = np.int32)
        
        if self._nb_classes == 2:
            fn = map_predicted_label_binary
        else:
            fn = map_predicted_label_multiclass

        predict = []
        for batch in predictions.iter_batches(batch_size = self.batch_size):
            predict.append(lambda : fn(batch, threshold))

        return np.concatenate(predict)



    """
    # This is a function for using with parent class training data decomposition that may be implemented later on
    def _prob_2_cls_multiclass(self, pred_dct, nb_records, threshold):
        print('_prob_2_cls_multiclass')
        def map_predicted_label(df):
            predict = pd.DataFrame({
                'best_proba': [df['predictions'][i][np.argmax(df['predictions'][i])] for i in range(len(df))],
                'predicted_label': df["predictions"].map(lambda x: np.array(x).argmax())
            })
            return predict

        global_predict = pd.DataFrame({
            'predict_proba': np.zeros(nb_records, dtype=np.float32),
            'predict_cls': np.zeros(nb_records, dtype=np.int32),
        })

        for tax, local_predict in pred_dct.items():
            with parallel_backend('threading'):
                local_predict = Parallel(n_jobs=-1, prefer='threads', verbose=1)(
                    delayed(map_predicted_label)(batch) for batch in local_predict.iter_batches(batch_size = self.batch_size))
            local_predict = pd.concat(local_predict, ignore_index=True)
            global_predict.loc[global_predict['predict_proba'] < local_predict['best_proba'],'predict_cls'] = np.array(local_predict.loc[local_predict['best_proba'] > global_predict['predict_proba'], 'predicted_label'])
            global_predict.loc[global_predict['predict_proba'] < local_predict['best_proba'],'predict_proba'] = np.array(local_predict.loc[local_predict['best_proba'] > global_predict['predict_proba'], 'best_proba'])
        # global_predict.loc[global_predict['predict_proba'] < threshold, 'predict_cls'] = -1
    
        return np.array(global_predict['predict_cls'])
    """            
# Training/building function outside of the class as mentioned on the Ray discussion
# https://discuss.ray.io/t/statuscode-resource-exhausted/4379/16
################################################################################

# Data streaming in PipelineDataset for larger than memory data, should prevent OOM
# https://docs.ray.io/en/latest/ray-air/check-ingest.html#enabling-streaming-ingest
# Smaller nb of workers + bigger nb CPU_per_worker + smaller batch_size to avoid memory overload
# https://discuss.ray.io/t/ray-sgd-distributed-tensorflow/261/8

# train_func with DatasetPipeline for Training data only
def train_func(config):
    # Parameters
    batch_size = config.get('batch_size', 128)
    epochs = config.get('epochs', 10)
    size = config.get('size')
    nb_cls = config.get('nb_cls')
    model = config.get('model')

    # Model setup 
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = build_model(model, nb_cls, size)

    # Load data directly to workers instead of serializing it?
    train_data = session.get_dataset_shard('train')
    val_data = session.get_dataset_shard('validation')

    def to_tf_dataset(data, batch_size):
        def to_tensor_iterator():
            for batch in data.iter_tf_batches(
                batch_size=batch_size
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

    batch_val = to_tf_dataset(val_data, batch_size)

    # Fit the model on streaming data
    for epoch_train in train_data.iter_epochs(epochs):
        batch_train = to_tf_dataset(epoch_train, batch_size)
        history = model.fit(
            x = batch_train,
            validation_data = batch_val,
            callbacks = [Callback()],
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

