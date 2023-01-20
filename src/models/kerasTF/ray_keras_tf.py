import os
import ray
import warnings
import numpy as np
import pandas as pd

from glob import glob
from shutil import rmtree
from utils import zip_X_y

# Preprocessing
from models.ray_tensor_min_max import TensorMinMaxScaler
from models.kerasTF.ray_one_hot_tensor import OneHotTensorEncoder
from ray.data.preprocessors import BatchMapper, Concatenator, LabelEncoder, Chain, OneHotEncoder

# Parent class / models
from models.ray_utils import ModelsUtils
from models.kerasTF.build_neural_networks import *

# Training
import tensorflow as tf
from ray.air import session, Checkpoint
from ray.air.integrations.keras import Callback
from ray.air.config import ScalingConfig, CheckpointConfig, DatasetConfig
from ray.train.tensorflow import TensorflowTrainer, TensorflowCheckpoint, prepare_dataset_shard

# Tuning
from ray.air.config import RunConfig

# Predicting
from ray.train.tensorflow import TensorflowPredictor
from ray.train.batch_predictor import BatchPredictor

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
        self._tuner = None
        self._nb_CPU_per_worker = 0
        # Computing variables
        if len(tf.config.list_physical_devices('GPU')) > 0:
            self._use_gpu = True
            self._n_workers = len(tf.config.list_physical_devices('GPU'))
            self._nb_CPU_per_worker = int((os.cpu_count()*0.8) / self._n_workers)
        else:
            self._use_gpu = False
            if os.cpu_count() > 4:
                if int(os.cpu_count()*0.8) % 5 == 0:
                    self._nb_CPU_per_worker = 5
                else:
                    self._nb_CPU_per_worker = 3
                self._n_workers = int((os.cpu_count()*0.8)/self._nb_CPU_per_worker)
            else:
                self._n_workers = 2
                self._nb_CPU_per_worker = 1

    
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

    def _fit_model(self, datasets):
        print('_fit_model')
        for name, ds in datasets.items():
            print(f'dataset preprocessing : {name}')
            ds = self._preprocessor.transform(ds)
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
                    use_stream_api = True
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

    def predict(self, df, threshold = 0.8, cv = False):
        print('predict')
        if df.count() > 0:
            df = df.window(blocks_per_window = 1)
            if len(df.schema().names) > 1:
                col_2_drop = [col for col in df.schema().names if col != '__value__']
                df = df.drop_columns(col_2_drop)

            df = self._preprocessor.preprocessors[0].transform(df)
            # Define predictor
            self._predictor = BatchPredictor.from_checkpoint(
                self._model_ckpt,
                TensorflowPredictor,
                model_definition = lambda : build_model(self.classifier, self._nb_classes, len(self.kmers))
            )
            # Make predictions
            predictions = self._predictor.predict(
                data=df,
                batch_size=self.batch_size
            )
            predictions = self._prob_2_cls(predictions, threshold)

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
        predict = np.ravel(np.array(predict.to_pandas()))

        return predict

# Training/building function outside of the class as mentioned on the Ray discussion
# https://discuss.ray.io/t/statuscode-resource-exhausted/4379/16
################################################################################

# Data streaming in PipelineDataset for larger than memory data, should prevent OOM
# https://docs.ray.io/en/latest/ray-air/check-ingest.html#enabling-streaming-ingest
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

    def to_tf_dataset(data):
        ds = tf.data.Dataset.from_tensors((
        tf.convert_to_tensor(list(data['__value__'])),
        tf.convert_to_tensor(list(data['labels']))
        ))
        return ds

    # Fit the model on streaming data
    results = []
    batch_val = pd.DataFrame(columns = ['__value__', 'labels'])
    for epoch in val_data.iter_epochs(1):
        for batch in epoch.iter_batches():
            batch_val = pd.concat([batch_val,batch])
    batch_val = to_tf_dataset(batch_val)

    report = {
        'accuracy' : 0,
        'loss' : 1000,
        'val_accuracy' : 0,
        'val_loss' : 1000
    }
    ckpt = None

    for epoch_train in train_data.iter_epochs(epochs):
        for batch_train in epoch_train.iter_batches():
            batch_train = to_tf_dataset(batch_train)
            history = model.fit(
                batch_train,
                validation_data = batch_val,
                callbacks=[Callback()],
                verbose=0
            )
            if history.history['val_accuracy'][0] > report['val_accuracy'] and history.history['val_loss'][0] < report['val_loss']:
                report['accuracy'] = history.history['accuracy'][0]
                report['loss'] = history.history['loss'][0]
                report['val_accuracy'] = history.history['val_accuracy'][0]
                report['val_loss'] = history.history['val_loss'][0]
                ckpt = TensorflowCheckpoint.from_model(model)
                results = [history.history]

    session.report(
        report,
        checkpoint = ckpt
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
