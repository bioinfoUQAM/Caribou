import os
import ray
import warnings
import numpy as np
import pandas as pd

# Parent class
from models.ray_utils import ModelsUtils

__author__ = 'Nicolas de Montigny'

__all__ = ['ModelsUtils','SklearnModel','KerasTFModel','BatchInferModel']

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
        super().__init__(classifier, outdir_results, batch_size, training_epochs, k, taxa, kmers_list, verbose)
        # Parameters
        self.dataset = dataset
        self.outdir_model = outdir_model
        if classifier in ['attention','lstm','deeplstm']:
            self.clf_file = '{}bacteria_binary_classifier_K{}_{}_{}_model'.format(outdir_model, k, classifier, dataset)
        else:
            self.clf_file = '{}{}_multiclass_classifier_K{}_{}_{}_model'.format(outdir_model, taxa, k, classifier, dataset)
        # # Initialize empty
        self.nb_classes = None
        # Variables for training with Ray
        self._strategy = distribute.MultiWorkerMirroredStrategy()
        if len(list_physical_devices('GPU')) > 0:
            self._trainer = Trainer(backend = 'tensorflow', num_workers = len(list_physical_devices('GPU')), use_gpu = True)
        else:
            self._trainer = Trainer(backend = 'tensorflow', num_workers = os.cpu_count())

    def _build(self, nb_kmers, nb_classes):
        print('_build')
        with self._strategy.scope():
            if self.classifier == 'attention':
                if self.verbose:
                    print('Training bacterial / host classifier based on Attention Weighted Neural Network')
                self._clf = build_attention(self.batch_size, self.k, nb_kmers)
            elif self.classifier == 'lstm':
                if self.verbose:
                    print('Training bacterial / host classifier based on Shallow LSTM Neural Network')
                self._clf = build_LSTM(self.k, self.batch_size)
            elif self.classifier == 'deeplstm':
                if self.verbose:
                    print('Training bacterial / host classifier based on Deep LSTM Neural Network')
                self._clf = build_deepLSTM(self.k, self.batch_size)
            elif self.classifier == 'lstm_attention':
                if self.verbose:
                    print('Training multiclass classifier based on Deep Neural Network hybrid between LSTM and Attention')
                self._clf = build_LSTM_attention(nb_kmers, nb_classes, self.batch_size)
            elif self.classifier == 'cnn':
                if self.verbose:
                    print('Training multiclass classifier based on CNN Neural Network')
                self._clf = build_CNN(self.k, self.batch_size, self.nb_classes)
            elif self.classifier == 'widecnn':
                if self.verbose:
                    print('Training multiclass classifier based on Wide CNN Network')
                self._clf = build_wideCNN(self.k, self.batch_size, self.nb_classes)

    def _fit_model(self, X, y):
        print('_fit_model')
        X = self._preprocess(X)
        y = self._label_encode(y)
        self.nb_classes = len(self.labels_map)

        checkpoint_strategy = CheckpointStrategy(num_to_keep = 1,
                                    checkpoint_score_attribute='val_accuracy',
                                    checkpoint_score_order='max')

        self._trainer.start()
        self._trainer.run(self._train_func, config = {'X':X,'y':y,'batch_size':self.batch_size,'epochs':self._training_epochs,'ids':self._ids_list,'nb_kmers':self._nb_kmers,'nb_classes':self.nb_classes}, checkpoint_strategy = checkpoint_strategy)
        self.checkpoint = self._trainer.best_checkpoint
        self._trainer.shutdown()

    def _train_func(self, config):
        print('_train_func')
        tf_config = json.loads(os.environ['TF_CONFIG'])
        num_workers = len(tf_config['cluster']['worker'])
        global_batch_size = config['batch_size'] * num_workers
        multi_worker_dataset = self._join_shuffle_data(config['X'], config['y'], global_batch_size, config['ids'], config['nb_kmers'])
        self._build(config['nb_kmers'], config['nb_classes'])
        early = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
        history = self._clf.fit(multi_worker_dataset, epochs = config['epochs'])
        print(history.history)
        save_checkpoint(model_weights = self._clf.get_weights())

    def _join_shuffle_data(self, X_train, y_train, batch_size, ids_list, nb_kmers):
        print('_join_shuffle_data')
        # Join
        X_train = X_train.to_pandas()
        y_train = y_train.to_pandas()
        X_train['id'] = ids_list
        y_train['id'] = ids_list
        df = X_train.merge(y_train, on = 'id', how = 'left')
        df = df.drop('id', 1)
        df = ray.data.from_pandas(df)
        df = df.random_shuffle()
        df = df.to_tf(
        label_column = self.taxa,
        batch_size = batch_size,
        output_signature = (
            TensorSpec(shape=(None, nb_kmers), dtype=int64),
            TensorSpec(shape=(None,), dtype=int64),))

        return df

    def predict(self, df, threshold = 0.8):
        print('predict')
        y_pred = pd.DataFrame(columns = ['id','classes'])
        y_pred['id'] = df.to_pandas().index()
        df = self._preprocess(df)
        if self.classifier in ['attention','lstm','deeplstm']:
            y_pred['classes'] = self._predict_binary(df)
        elif self.classifier in ['lstm_attention','cnn','widecnn']:
            y_pred['classes'] = self._predict_multi(df, threshold)

        return y_pred

    def _predict_binary(self, df):
        predictor = TensorflowPredictor().from_checkpoint(checkpoint = self.checkpoint, model_definition = self._build)

        predicted = np.array(predictor.predict(df))

        y_pred = np.around(predicted.reshape(1, predicted.size)[0]).astype(np.int64)

        return self._label_decode(y_pred)

    def _predict_multi(self, df, threshold):
        y_pred = np.empty(df.count(), dtype=np.int32)
        predictor = TensorflowPredictor().from_checkpoint(checkpoint = self.checkpoint, model_definition = self._build)

        predicted = np.array(predictor.predict(df))

        for i in range(len(predicted)):
            if np.isnan(predicted[i,np.argmax(predicted[i])]):
                y_pred[i] = -1
            elif predict[i,np.argmax(predicted[i])] >= threshold:
                y_pred[i] = np.argmax(predicted[i])
            else:
                y_pred[i] = -1

        predictions = self._label_threshold(predictions, threshold)

        return self._label_decode(y_pred)

    def __reduce__(self):
        deserializer = self.__class__
        serialized_data = (self.classifier, self.dataset, self.outdir_model, self.outdir_results, self.batch_size, self._training_epochs, self.k, self.taxa, self.verbose)

        return deserializer, serialized_data

    def _label_threshold(self, arr, threshold):
        arr = np.array(arr.to_pandas())
        nb_labels = len(arr)
        predict = np.empty(nb_labels, dtype = np.int32)
        for i in range(nb_labels):
            if np.isnan(arr[i,np.argmax(arr[i])]):
                predict[i] = -1
            elif arr[i,np.argmax(arr[i])] >= threshold:
                predict[i] = np.argmax(arr[i])
            else:
                predict[i] = -1
        return ray.data.from_numpy(predict)
