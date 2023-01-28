import os
import ray
import warnings
import numpy as np
import pandas as pd

from glob import glob
from utils import zip_X_y
from shutil import rmtree

# Preprocessing
from models.ray_tensor_min_max import TensorMinMaxScaler
from models.kerasTF.ray_one_hot_tensor import OneHotTensorEncoder
from ray.data.preprocessors import LabelEncoder, Chain

# Parent class / models
from models.ray_utils import ModelsUtils
from models.kerasTF.build_neural_networks import *

# Training
import tensorflow as tf
from ray.air import session, Checkpoint
from ray.air.integrations.keras import Callback
from ray.air.config import ScalingConfig, DatasetConfig
from ray.train.tensorflow import TensorflowTrainer, TensorflowCheckpoint, prepare_dataset_shard

# Tuning
from ray.air.config import RunConfig

# Predicting
from ray.train.tensorflow import TensorflowPredictor
from ray.train.batch_predictor import BatchPredictor
from joblib import Parallel, delayed, parallel_backend

# Simulation class
from models.reads_simulation import readsSimulation

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
            batch_size, # Must be small to reduce mem usage
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
        self._nb_CPU_per_worker = 0
        # Computing variables
        if len(tf.config.list_physical_devices('GPU')) > 0:
            self._use_gpu = True
            self._n_workers = len(tf.config.list_physical_devices('GPU'))
            self._nb_CPU_per_worker = int((os.cpu_count()*0.8) / self._n_workers)
        else:
            self._use_gpu = False
            self._n_workers = 1
            self._nb_CPU_per_worker = int((os.cpu_count()*0.8) - 1)

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
            TensorMinMaxScaler(self.kmers),
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

        return decoded

    def train(self, datasets, kmers_ds, cv = True):
        print('train')

        df = datasets['train']
        
        if cv:
            df_test = datasets['test']
            self._cross_validation(df, df_test, kmers_ds)
        else:
            df_train, df_val = df.train_test_split(0.2, shuffle = True)
            df_val = self._sim_4_val(df_val, kmers_ds, 'validation')
            df_train = df_train.drop_columns(['id'])
            df_val = df_val.drop_columns(['id'])
            datasets = {'train': df_train, 'validation': df_val}
            self._fit_model(datasets)

    def _sim_4_val(self, df, kmers_ds, name):
        sim_genomes = []
        sim_taxas = []
        for row in df.iter_rows():
            sim_genomes.append(row['id'])
            sim_taxas.append(row[self.taxa])
        cls = pd.DataFrame({'id':sim_genomes,self.taxa:sim_taxas})
        sim_outdir = os.path.dirname(kmers_ds['profile'])
        cv_sim = readsSimulation(kmers_ds['fasta'], cls, sim_genomes, 'miseq', sim_outdir, name)
        sim_data = cv_sim.simulation(self.k, self.kmers)
        sim_ids = sim_data['ids']
        sim_ids = sim_data['ids']
        sim_cls = pd.DataFrame({'sim_id':sim_ids}, dtype = object)
        sim_cls['id'] = sim_cls['sim_id'].str.replace('_[0-9]+_[0-9]+_[0-9]+', '', regex=True)
        sim_cls = sim_cls.set_index('id').join(cls.set_index('id'))
        sim_cls = sim_cls.drop(['sim_id'], axis=1)
        sim_cls = sim_cls.reset_index(drop = True)
        df = ray.data.read_parquet(sim_data['profile'])
        df = zip_X_y(df, sim_cls)
        return df

    def _cross_validation(self, df_train, df_test, kmers_ds):
        print('_cross_validation')

        df_train, df_val = df_train.train_test_split(0.2, shuffle = True)
        
        df_val = self._sim_4_val(df_val, kmers_ds, '{}_val'.format(self.dataset))
        
        df_train = df_train.drop_columns(['id'])
        df_test = df_test.drop_columns(['id'])
        df_val = df_val.drop_columns(['id'])

        datasets = {'train' : df_train, 'validation' : df_val}
        self._fit_model(datasets)

        y_true = []
        for row in df_test.iter_rows():
            y_true.append(row[self.taxa])

        y_pred = self.predict(df_test.drop_columns([self.taxa]), cv = True)

        for file in glob(os.path.join(os.path.dirname(kmers_ds['profile']), '*sim*')):
            if os.path.isdir(file):
                rmtree(file)
            else:
                os.remove(file)

        self._cv_score(y_true, y_pred)

    """
    # Model training with generators
    # Based on https://docs.ray.io/en/latest/ray-air/examples/torch_incremental_learning.html
    def _fit_model(self, datasets):
        print('_fit_model')
        
        train_stream = datasets['train'].window(blocks_per_window = 10)
        train_stream = self._preprocessor.transform(train_stream)
        val_dataset = self._preprocessor.transform(datasets['validation']).fully_executed()

        # Training parameters
        self._train_params = {
            'batch_size': self.batch_size,
            'epochs': self._training_epochs,
            'size': self._nb_kmers,
            'nb_cls':self._nb_classes,
            'classifier': self.classifier
        }

        print(f'num_workers : {self._n_workers}')
        print(f'nb_CPU_per_worker : {self._nb_CPU_per_worker}')
        
        latest_checkpoint = None

        accuracy_for_all_tasks = []
        task_idx = 0
        all_checkpoints = []

        for train_dataset in train_stream.iter_datasets():
            print(train_dataset)

            # Define trainer / tuner
            self._trainer = TensorflowTrainer(
                train_loop_per_worker=train_func,
                train_loop_config=self._train_params,
                scaling_config=ScalingConfig(
                    trainer_resources={'CPU': 1},
                    num_workers=self._n_workers,
                    use_gpu=self._use_gpu,
                    resources_per_worker={'CPU': self._nb_CPU_per_worker}
                ),
                dataset_config={
                    'train': DatasetConfig(
                        fit=False,
                        transform=False,
                        split=False,
                        use_stream_api=False
                    )
                },
                run_config=RunConfig(
                    name=self.classifier,
                    local_dir=self._workdir,
                ),
                datasets={'train': train_dataset},
                resume_from_checkpoint=latest_checkpoint
            )
            # Train / tune execution
            training_result = self._trainer.fit()
            latest_checkpoint = training_result.best_checkpoints[0][0]

            correct_dataset = batch_predict_val(
                latest_checkpoint,
                val_dataset,
                self.classifier,
                self.batch_size,
                self._nb_classes,
                len(self.kmers)
            )
            
            accuracy_this_task = correct_dataset.sum(on="correct") / correct_dataset.count()
            accuracy_for_all_tasks.append(accuracy_this_task)
            all_checkpoints.append(latest_checkpoint)
        
        best_accuracy_pos = accuracy_for_all_tasks.index(np.argmax(accuracy_for_all_tasks))
        self._model_ckpt = all_checkpoints[best_accuracy_pos]
    """

    # Model training with DatasetPipeline
    def _fit_model(self, datasets):
        print('_fit_model')
        for name, ds in datasets.items():
            print(f'dataset preprocessing : {name}')
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

        print(f'num_workers : {self._n_workers}')
        print(f'nb_CPU_per_worker : {self._nb_CPU_per_worker}')

        # Define trainer / tuner
        self._trainer = TensorflowTrainer(
            train_loop_per_worker = train_func,
            train_loop_config = self._train_params,
            scaling_config = ScalingConfig(
                trainer_resources={'CPU': 1},
                num_workers = self._n_workers,
                use_gpu = self._use_gpu,
                resources_per_worker={'CPU': self._nb_CPU_per_worker}
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
            datasets = datasets,
        )
        # Train / tune execution
        training_result = self._trainer.fit()
        self._model_ckpt = training_result.best_checkpoints[0][0]
    

    def predict(self, df, threshold=0.8, cv=False):
        print('predict')
        if df.count() > 0:
            if len(df.schema().names) > 1:
                col_2_drop = [col for col in df.schema().names if col != '__value__']
                df = df.drop_columns(col_2_drop)

            # Preprocess
            df = self._preprocessor.preprocessors[0].transform(df)

            # Make predictions
            predictions = batch_prediction(
                self._model_ckpt,
                df,
                self.classifier,
                self.batch_size,
                self._nb_classes,
                len(self.kmers)
            )

            # Convert predictions to labels
            predictions = self._prob_2_cls(predictions, threshold)

            return self._label_decode(predictions)
        else:
            raise ValueError('No data to predict')

    """
    def predict(self, df, threshold = 0.8, cv = False):
        print('predict')
        if df.count() > 0:
            # df = df.window(blocks_per_window = 1)
            print('col_2_drop')
            if len(df.schema().names) > 1:
                col_2_drop = [col for col in df.schema().names if col != '__value__']
                df = df.drop_columns(col_2_drop)

            df = self._preprocessor.preprocessors[0].transform(df)
            # Define predictor
            print('BatchPredictor.from_checkpoint')
            self._predictor = BatchPredictor.from_checkpoint(
                self._model_ckpt,
                TensorflowPredictor,
                model_definition = lambda : build_model(self.classifier, self._nb_classes, len(self.kmers))
            )
            # Make predictions
            print('predict')
            predictions = self._predictor.predict(
                data = df,
                batch_size = self.batch_size,
            )

            # Make predictions
            # print('predict_pipelined')
            # predictions = self._predictor.predict_pipelined(
            #     data = df,
            #     bytes_per_window = 50000000,
            #     batch_size = self.batch_size,
            # )

            predictions = self._prob_2_cls(predictions, threshold)

            return self._label_decode(predictions)
        else:
            raise ValueError('No data to predict')
    """
    # Iterate over batches of predictions to transform probabilities to labels without mapping
    def _prob_2_cls(self, predictions, threshold):
        print('_prob_2_cls')
        def map_predicted_label_binary(df, threshold):
            lower_threshold = 0.5 - (threshold * 0.5)
            upper_threshold = 0.5 + (threshold * 0.5)
            predict = pd.DataFrame({
                'proba': df['predictions'],
                'predicted_label': np.full(len(df), -1)
            })
            predict.loc[predict['proba'] >= upper_threshold, 'predicted_label'] = 1
            predict.loc[predict['proba'] <= lower_threshold, 'predicted_label'] = 0
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

        with parallel_backend('threading'):
            predict = Parallel(n_jobs=-1, prefer='threads', verbose=1)(
                delayed(fn)(batch, threshold) for batch in predictions.iter_batches(batch_size = self.batch_size))

        return np.concatenate(predict)
                
# Training/building function outside of the class as mentioned on the Ray discussion
# https://discuss.ray.io/t/statuscode-resource-exhausted/4379/16
################################################################################

# Data streaming in PipelineDataset for larger than memory data, should prevent OOM
# https://docs.ray.io/en/latest/ray-air/check-ingest.html#enabling-streaming-ingest
# Smaller nb of workers + bigger nb CPU_per_worker + smaller batch_size to avoid memory overload
# https://discuss.ray.io/t/ray-sgd-distributed-tensorflow/261/8
"""
# train_func with Training data only
def train_func(config):
    batch_size = config.get('batch_size', 128)
    epochs = config.get('epochs', 10)
    size = config.get('size')
    nb_cls = config.get('nb_cls')
    classifier = config.get('classifier')

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

    # Model setup
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = build_model(classifier, nb_cls, size)
        checkpoint = session.get_checkpoint()
        if checkpoint:
            checkpoint_dict = checkpoint.to_dict()
            model.set_weights(checkpoint_dict.get("model_weights"))
        if classifier in ['attention','lstm','deeplstm']:
            model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        else:
            model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    train_data = session.get_dataset_shard('train')
    train_data = to_tf_dataset(train_data, batch_size)

    history = model.fit(
            train_data,
            epochs = epochs,
            callbacks=[Callback()],
            verbose=0
    )
    session.report({
        'accuracy': history.history['accuracy'][0],
        'loss': history.history['loss'][0]
    },
        checkpoint=Checkpoint.from_dict(dict(model_weights=model.get_weights()))
    )
"""

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
    results = []

    for epoch_train in train_data.iter_epochs(epochs):
        batch_train = to_tf_dataset(epoch_train, batch_size)
        history = model.fit(
                x=batch_train,
                validation_data=batch_val,
                callbacks=[Callback()],
                verbose=0
        )
        results.append(history.history)
        session.report({
            'accuracy': history.history['accuracy'][0],
            'loss': history.history['loss'][0],
            'val_accuracy': history.history['val_accuracy'][0],
            'val_loss': history.history['val_loss'][0],
        },
            checkpoint=TensorflowCheckpoint.from_model(model)
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

def batch_predict_val(checkpoint, batch, clf, batch_size, nb_classes, nb_kmers):
    def convert_logits_to_classes(df):
        best_class = df["predictions"].map(lambda x: np.array(x).argmax())
        df["predictions"] = best_class
        return df

    def calculate_prediction_scores(df):
            return pd.DataFrame({"correct": df["predictions"] == df["labels"]})

    predictor = BatchPredictor.from_checkpoint(
        checkpoint,
        TensorflowPredictor,
        model_definition = lambda: build_model(clf, nb_classes, nb_kmers)
    )
    predictions = predictor.predict(
        data = batch,
        batch_size = batch_size,
        feature_columns = ['__value__'],
        keep_columns = ['labels']
    )
    pred_results = predictions.map_batches(
        convert_logits_to_classes,
        batch_format="pandas"
    )
    correct_dataset = pred_results.map_batches(
        calculate_prediction_scores,
        batch_format="pandas",
    )
    
    return correct_dataset

def batch_prediction(checkpoint, batch, clf, batch_size, nb_classes, nb_kmers):
    predictor = BatchPredictor.from_checkpoint(
        checkpoint,
        TensorflowPredictor,
        model_definition = lambda: build_model(clf, nb_classes, nb_kmers)
    )
    predictions = predictor.predict(
        data = batch,
        batch_size = batch_size
    )
    return predictions
