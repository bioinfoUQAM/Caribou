import os
import ray
import warnings
import numpy as np
import pandas as pd

# Preprocessing
from models.encoders.model_label_encoder import ModelLabelEncoder
from models.encoders.onesvm_label_encoder import OneClassSVMLabelEncoder
from models.preprocessors.tfidf_transformer import TensorTfIdfTransformer

# Training
import ray.cloudpickle as cpickle
from ray.air.config import ScalingConfig
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from models.sklearn.partial_trainer import SklearnPartialTrainer
from models.sklearn.scoring_one_svm import ScoringSGDOneClassSVM

# Tuning
from ray.air.config import RunConfig

# Predicting
from ray.train.batch_predictor import BatchPredictor
from models.sklearn.tensor_predictor import SklearnTensorPredictor
from models.sklearn.probability_predictor import SklearnTensorProbaPredictor

# Parent classes
from models.sklearn.models import SklearnModels
from models.multiclass_utils import MulticlassUtils

# Data
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

TENSOR_COLUMN_NAME = '__value__'
LABELS_COLUMN_NAME = 'labels'

__author__ = 'Nicolas de Montigny'

__all__ = ['SklearnModel']

# Ignore warnings to have a more comprehensible output on stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

class SklearnMulticlassModels(SklearnModels, MulticlassUtils):
    """
    Class used to build, train and predict multiclass models using Ray with Scikit-learn backend

    ----------
    Attributes
    ----------

    clf_file : string
        Path to a file containing the trained model for this object

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
        self._training_collection = {}
        self._encoder = {}
        self._trainer = {}
        self._model_ckpt = {}
        self._predictor = {}

    def preprocess(self, ds, scaling = False, scaler_file = None):
        print('preprocess')

        if scaling:
            self._scaler = TensorTfIdfTransformer(self.kmers, scaler_file)
            self._scaler.fit(ds)
        
        self._encoder = ModelLabelEncoder(self.taxa)
        self._encoder.fit(ds)

        # Labels mapping
        labels = list(self._encoder.stats_[f'unique_values({self.taxa})'].keys())
        encoded = np.arange(len(labels))
        labels = np.append(labels, 'Unknown')
        encoded = np.append(encoded, -1)
        
        self._labels_map = {}
        for (label, encode) in zip(labels, encoded):
            self._labels_map[label] = encode
        
        # self._weights = self._compute_weights()

    def fit(self, datasets):
        print('fit')
    # TODO: remove validation from datasets
    # train / val on training ds, CV on test ds
        ds = datasets['train']
        ds = ds.drop_columns(['id'])
        ds = self._encoder.transform(ds)
        if self._scaler is not None:
            ds = self._scaler.transform(ds)

        # One sub-model per artificial cluster of samples
        ds = self._random_split_dataset(ds)
        # checkpointing directory
        model_dir = os.path.join(self._workdir, self.classifier)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        # Model-specific training functions
        def build_fit_sgd(data):
            X = data[TENSOR_COLUMN_NAME]
            y = data[LABELS_COLUMN_NAME]
            prev_label = data['cluster'][0]
            model = SGDClassifier(
                alpha = 173.5667373,
                learning_rate = 'optimal',
                loss = 'modified_huber',
                penalty = 'l2',
                # 'class_weight' : self._weights,
            )
            model.fit(X, y)

            model_file = os.path.join(model_dir, f'{prev_label}.pkl')

            with open(model_file, "wb") as file:
                cpickle.dump(model, file)

            return {
                'cluster' : [prev_label],
                'file' : [model_file]
            }

        def build_fit_mnb(data):
            X = data[TENSOR_COLUMN_NAME]
            y = data[LABELS_COLUMN_NAME]
            prev_label = data['cluster'][0]
            model = SGDClassifier(
                alpha = 173.5667373,
                learning_rate = 'optimal',
                loss = 'modified_huber',
                penalty = 'l2',
                # 'class_weight' : self._weights,
            )
            model.fit(X, y)

            model_file = os.path.join(model_dir, f'{prev_label}.pkl')

            with open(model_file, "wb") as file:
                cpickle.dump(model, file)

            return {
                'cluster' : [prev_label],
                'file' : [model_file]
            }
        
        if self.classifier == 'sgd':
            print('Training multiclass SGD classifier')
            training_result = ds.map_groups(build_fit_sgd, batch_format = 'numpy')
        elif self.classifier == 'mnb':
            print('Training multiclass Multinomial Naive Bayes classifier')
            training_result = ds.map_groups(build_fit_mnb, batch_format = 'numpy')

        training_result = training_result.to_pandas().to_dict('records')
        for record in training_result:
            self._model_ckpt[record['cluster']] = record['file']
        
    def predict(self, ds):
        print('predict')
        probabilities = self._predict_proba(ds)
        predictions = np.argmax(probabilities, axis = 1)
        predictions = self._label_decode(predictions)
        return predictions
    
    def predict_proba(self, ds, threshold = 0.8):
        print('predict_proba')
        probabilities = self._predict_proba(ds)
        predictions = self._get_threshold_pred(probabilities, threshold)
        return self._label_decode(predictions)

    def _predict_proba(self, ds):
        if ds.count() > 0:
            if self._scaler is not None:
                ds = self._scaler.transform(ds)
            # ds = ds.materialize()

            def predict_func(data):
                X = _unwrap_ndarray_object_type_if_needed(data[TENSOR_COLUMN_NAME])
                pred = np.zeros((len(X), len(self._labels_map)))
                for cluster, model_file in self._model_ckpt.items():
                    with open(model_file, 'rb') as file:
                        model = cpickle.load(file)
                    proba = model.predict_proba(X)
                    for i, cls in enumerate(model.classes_):
                        pred[:, cls] += proba[:, i]
                pred = pred / len(self._model_ckpt)
                return {'predictions' : pred}

            probabilities = ds.map_batches(predict_func, batch_format = 'numpy')
            probabilities = _unwrap_ndarray_object_type_if_needed(probabilities.to_pandas()['predictions'])

        return probabilities

    def _get_threshold_pred(self, predict, threshold):
        print('_get_threshold_pred')
        proba_predict = {
            'best_proba' : [],
            'predicted_label' : []
        }
        for line in predict:
            proba_predict['best_proba'].append(line[np.argmax(line)]),
            proba_predict['predicted_label'].append(np.argmax(line))

        proba_predict = pd.DataFrame(proba_predict)
        proba_predict.loc[proba_predict['best_proba'] < threshold, 'predicted_label'] = -1

        return proba_predict['predicted_label']
    
    def _label_decode(self, predict):
        print('_label_decode')
        decoded = pd.Series(np.empty(len(predict), dtype=object))
        for label, encoded in self._labels_map.items():
            decoded[predict == encoded] = label

        return np.array(decoded)
