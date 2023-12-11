import os
import ray
import warnings
import numpy as np
import pandas as pd

# Preprocessing
from models.encoders.model_label_encoder import ModelLabelEncoder
from models.preprocessors.min_max_scaler import TensorMinMaxScaler
from models.encoders.onesvm_label_encoder import OneClassSVMLabelEncoder
from models.preprocessors.tfidf_transformer import TensorTfIdfTransformer

# Training
import ray.cloudpickle as cpickle
from ray.air.config import ScalingConfig
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
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

    # Data preprocessing
    #########################################################################################################

    def preprocess(self, ds, scaling = False, scaler_file = None):
        print('preprocess')
        # Labels encoding
        self._encoder = ModelLabelEncoder(self.taxa)
        self._encoder.fit(ds)

        # Labels mapping
        labels = list(self._encoder.stats_[f'unique_values({self.taxa})'].keys())
        encoded = np.arange(len(labels))
        labels = np.append(labels, 'Unknown')
        encoded = np.append(encoded, -1)
        
        for (label, encode) in zip(labels, encoded):
            self._labels_map[label] = encode
        
        # Class weights
        self._weights = self._compute_weights()
        
        # Scaling
        self._scaler = TensorMinMaxScaler(self._nb_kmers)
        self._scaler.fit(ds)
        
    # Models training
    #########################################################################################################

    def fit(self, datasets):
        print('fit')
        # for name, ds in datasets.items():
            # ds = ds.drop_columns(['id'])
        train_ds = datasets['train']
        train_ds = self._encoder.transform(train_ds)
        train_ds = self._scaler.transform(train_ds)
        # datasets[name] = ds

        # One sub-model per artificial cluster of samples
        train_ds = self._random_split_dataset(train_ds)
        # val_ds = datasets['validation'].to_pandas()
        
        # Checkpointing directory
        model_dir = os.path.join(self._workdir, f'{self.classifier}_{self.taxa}')
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        # Model-specific training functions
        def build_fit_sgd(train_data):#, val_data):
            # Training data
            X_train = _unwrap_ndarray_object_type_if_needed(train_data[TENSOR_COLUMN_NAME])
            y_train = np.array(train_data[LABELS_COLUMN_NAME])
            # Validation data
            # X_val = val_data[TENSOR_COLUMN_NAME]
            # y_val = val_data[LABELS_COLUMN_NAME]
            # msk_val = y_val.isin(np.unique(y_train))
            # X_val = _unwrap_ndarray_object_type_if_needed(X_val[msk_val])
            # y_val = np.array(y_val[msk_val])
            cluster = train_data['cluster'][0]
            model = SGDClassifier(
                learning_rate = 'optimal',
                loss = 'modified_huber',
                penalty = 'l2',
                class_weight = self._weights,
            )
            model.fit(X_train, y_train)

            # calibrator = CalibratedClassifierCV(
            #     estimator = model,
            #     method = 'isotonic',
            #     cv = 'prefit',     
            # )

            # calibrator.fit(X_val,y_val)

            model_file = os.path.join(model_dir, f'{cluster}.pkl')

            with open(model_file, "wb") as file:
                cpickle.dump(model, file)

            return {
                'cluster' : [cluster],
                'file' : [model_file]
            }

        def build_fit_mnb(train_data):#, val_data):
            # Training data
            X_train = _unwrap_ndarray_object_type_if_needed(train_data[TENSOR_COLUMN_NAME])
            y_train = np.array(train_data[LABELS_COLUMN_NAME])
            # Validation data
            # X_val = val_data[TENSOR_COLUMN_NAME]
            # y_val = val_data[LABELS_COLUMN_NAME]
            # msk_val = y_val.isin(np.unique(y_train))
            # X_val = _unwrap_ndarray_object_type_if_needed(X_val[msk_val])
            # y_val = np.array(y_val[msk_val])
            cluster = train_data['cluster'][0]
            model = MultinomialNB()
            model.fit(X_train, y_train)

            model_file = os.path.join(model_dir, f'{cluster}.pkl')

            # calibrator = CalibratedClassifierCV(
            #     estimator = model,
            #     method = 'isotonic',
            #     cv = 'prefit',     
            # )

            # calibrator.fit(X_val,y_val)

            with open(model_file, "wb") as file:
                cpickle.dump(model, file)

            return {
                'cluster' : [cluster],
                'file' : [model_file]
            }
        
        if self.classifier == 'sgd':
            print('Training multiclass SGD classifier')
            training_result = train_ds.map_groups(build_fit_sgd, batch_format = 'numpy')
            # training_result = train_ds.map_groups(lambda ds: build_fit_sgd(ds, val_ds), batch_format = 'numpy')
        elif self.classifier == 'mnb':
            print('Training multiclass Multinomial Naive Bayes classifier')
            training_result = train_ds.map_groups(build_fit_mnb, batch_format = 'numpy')
            # training_result = train_ds.map_groups(lambda ds: build_fit_mnb(ds, val_ds), batch_format = 'numpy')

        training_result = training_result.to_pandas().to_dict('records')
        for record in training_result:
            self._model_ckpt[record['cluster']] = record['file']
        
    # Models predicting
    #########################################################################################################

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
                # pred = pred / len(self._model_ckpt)
                return {'predictions' : pred}

            probabilities = ds.map_batches(predict_func, batch_format = 'numpy')
            probabilities = _unwrap_ndarray_object_type_if_needed(probabilities.to_pandas()['predictions'])
            
            return probabilities
        else:
            raise ValueError('Empty dataset, cannot execute predictions!')

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
