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

# Parent class
from models.sklearn.models import SklearnModels

TENSOR_COLUMN_NAME = '__value__'
LABELS_COLUMN_NAME = 'labels'

__author__ = 'Nicolas de Montigny'

__all__ = ['SklearnModel']

# Ignore warnings to have a more comprehensible output on stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

class SklearnBinaryModels(SklearnModels):
    """
    Class used to build, train and predict binary models using Ray with Scikit-learn backend

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

    # Data preprocessing
    #########################################################################################################

    def preprocess(self, ds, scaling = False, scaler_file = None):
        print('preprocess')
        # Labels encoding + mapping
        if self.classifier == 'onesvm':
            self._encoder = OneClassSVMLabelEncoder(self.taxa)
            self._encoded = np.array([1,-1], dtype = np.int32)
            labels = np.array(['Bacteria', 'Unknown'], dtype = object)
            self._encoder.fit(ds)
        else:
            self._encoder = ModelLabelEncoder(self.taxa)
            self._encoder.fit(ds)
            labels = list(self._encoder.stats_[f'unique_values({self.taxa})'].keys())
            self._encoded = np.arange(len(labels))
            labels = np.append(labels, 'Unknown')
            self._encoded = np.append(self._encoded, -1)
            # Class weights
            self._weights = self._compute_weights()

        # Labels mapping
        for (label, encoded) in zip(labels, self._encoded):
            self._labels_map[label] = encoded

        # Scaling
        self._scaler = TensorTfIdfTransformer(
            features = self.kmers,
            file = scaler_file
        )
        # self._scaler = TensorMinMaxScaler(self._nb_kmers)
        self._scaler.fit(ds)


    # Model training
    #########################################################################################################

    def fit(self, datasets):
        print('_fit_model')
        # Define model
        self._build()
        for name, ds in datasets.items():
            # ds = ds.drop_columns(['id'])
            ds = self._encoder.transform(ds)
            ds = self._scaler.transform(ds)
            datasets[name] = ray.put(ds)
        
        try:
            training_labels = self._encoded.copy()
            training_labels = np.delete(training_labels, np.where(training_labels == -1))
        except:
            pass

        # Define trainer
        self._trainer = SklearnPartialTrainer(
            estimator=self._clf,
            labels_list=training_labels,
            features_list=self.kmers,
            params=self._train_params,
            datasets=datasets,
            batch_size=self.batch_size,
            training_epochs=self._training_epochs,
            set_estimator_cpus=True,
            scaling_config=ScalingConfig(
                trainer_resources={
                    'CPU': int(os.cpu_count()*0.6)
                }
            ),
            run_config=RunConfig(
                name=self.classifier,
                local_dir=self._workdir
            ),
        )

        # Training execution
        training_result = self._trainer.fit()
        self._model_ckpt = training_result.checkpoint
    
    def _build(self):
        print('_build')
        if self.classifier == 'onesvm':
            print('Training bacterial extractor with One Class SVM')
            self._clf = ScoringSGDOneClassSVM()
            self._train_params = {
                'learning_rate' : 'optimal'
            }
        else :
            print('Training bacterial / host classifier with SGD')
            self._clf = SGDClassifier()
            self._train_params = {
                'loss' : 'hinge',
                'learning_rate' : 'optimal',
                'class_weight' : self._weights,
                'n_jobs' : -1
            }

    # Model predicting
    #########################################################################################################

    def predict(self, ds):
        print('predict')
        if ds.count() > 0:
            ds = self._scaler.transform(ds)
            ds = ds.materialize()
            predict_kwargs = {'features':self.kmers, 'num_estimator_cpus':-1}
            self._predictor = BatchPredictor.from_checkpoint(self._model_ckpt, SklearnTensorPredictor)
            predictions = self._predictor.predict(ds, batch_size = self.batch_size, feature_columns = [TENSOR_COLUMN_NAME], **predict_kwargs)
            predictions = np.array(predictions.to_pandas()).reshape(-1)
            return self._label_decode(predictions)
        else:
            raise ValueError('No data to predict')
    
    def predict_proba(self, ds, threshold = 0.8):
        print('predict_proba')
        # No predict_proba methods implemented for these models
        return self.predict(ds)

    def _get_threshold_pred(self, predict, nb_cls, threshold):
        print('_get_threshold_pred')
        def map_predicted_label(ds : pd.DataFrame):
            predict = pd.DataFrame({
                'best_proba': [max(ds.iloc[i].values) for i in range(len(ds))],
                'predicted_label': [np.argmax(ds.iloc[i].values) for i in range(len(ds))]
            })
            predict.loc[predict['best_proba'] < threshold, 'predicted_label'] = -1
            return pd.DataFrame(predict['predicted_label'])

        if nb_cls == 1:
            predict = np.round(abs(np.concatenate(predict.to_pandas()['predictions'])))
        else:
            predict = predict.map_batches(map_predicted_label, batch_format = 'pandas')
            predict = np.ravel(np.array(predict.to_pandas()))

        return predict