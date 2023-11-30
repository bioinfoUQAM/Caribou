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
        
        self._training_collection = self._split_dataset(ds, self.taxa, self._csv)
        
        for prev_taxa, ds in self._training_collection.items():
            self._encoder[prev_taxa] = ModelLabelEncoder(self.taxa)
            self._encoder[prev_taxa].fit(ds)

            # Labels mapping
            labels = list(self._encoder[prev_taxa].stats_[f'unique_values({self.taxa})'].keys())
            encoded = np.arange(len(labels))
            labels = np.append(labels, 'Unknown')
            encoded = np.append(encoded, -1)
            
            self._labels_map[prev_taxa] = {}
            for (label, encode) in zip(labels, encoded):
                self._labels_map[prev_taxa][label] = encode
            
            # self._weights[prev_taxa] = self._compute_weights()

    def _build(self):
        print('_build')
# TODO: Test performances for classifiers, if need more accuracy -> sklearn.multiclass.OneVsRestClassifier for multiple binary problems
        # if self.classifier == 'sgd':
        print('Training multiclass SGD classifier')
        self._clf = SGDClassifier()
        self._train_params = {
            'alpha' : 173.5667373,
            'learning_rate' : 'optimal',
            'loss': 'modified_huber',
            'penalty' : 'l2',
            # 'class_weight' : self._weights,
        }
        # elif self.classifier == 'mnb':
        #     print('Training multiclass Multinomial Naive Bayes classifier')
        #     self._clf = MultinomialNB()
        #     self._train_params = {
        #         'alpha' : 0.243340248,
        #         'fit_prior' : True
        #     }

    def fit(self, datasets):
        print('_fit_model')
        # Define model
        self._build()
        training_result = {}
        for prev_taxa, ds in self._training_collection.items():
            ds = ds.drop_columns(['id'])
            ds = self._encoder.transform(ds)
            if self._scaler is not None:
                ds = self._scaler.transform(ds)
            # Trigger the preprocessing computations before ingest in trainer
            # Otherwise, it would be executed at each epoch
            ds = ds.materialize()
            datasets['train'] = ray.put(ds)

            try:
                training_labels = list(self._labels_map[prev_taxa].values())
                training_labels = np.delete(training_labels, np.where(training_labels == -1))
            except:
                pass

            # Define trainer
            self._trainer[prev_taxa] = SklearnPartialTrainer(
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
            training_result[prev_taxa] = self._trainer.fit()
            self._model_ckpt[prev_taxa] = training_result[prev_taxa].checkpoint
        
    def predict(self, ds):
        print('predict')
        if ds.count() > 0:
            if self._scaler is not None:
                ds = self._scaler.transform(ds)

            ds = ds.materialize()
            predict_kwargs = {'features':self.kmers, 'num_estimator_cpus':-1}
            
            for prev_taxa, ckpt in self._model_ckpt.items():
                self._predictor[prev_taxa] = BatchPredictor.from_checkpoint(ckpt, SklearnTensorProbaPredictor)
                predictions = self._predictor[prev_taxa].predict(ds, batch_size = self.batch_size, feature_columns = [TENSOR_COLUMN_NAME], **predict_kwargs)
            predictions = self._predictions_grouping(predictions)
            return self._label_decode(predictions)
        else:
            raise ValueError('No data to predict')
    
    def predict_proba(self, ds, threshold = 0.8):
        print('predict_proba')
        print('predict')
        if ds.count() > 0:
            if self._scaler is not None:
                ds = self._scaler.transform(ds)
            ds = ds.materialize()
            predict_kwargs = {'features':self.kmers, 'num_estimator_cpus':-1}
            self._predictor = BatchPredictor.from_checkpoint(self._model_ckpt, SklearnTensorProbaPredictor)
            predictions = self._predictor.predict(ds, batch_size = self.batch_size, feature_columns = [TENSOR_COLUMN_NAME], **predict_kwargs)
            predictions = np.array(predictions.to_pandas()).reshape(-1)
            return self._label_decode(predictions)
        else:
            raise ValueError('No data to predict')

    def _get_threshold_pred(self, predict, threshold):
        print('_get_threshold_pred')
        def map_predicted_label(ds : pd.DataFrame):
            predict = pd.DataFrame({
                'best_proba': [max(ds.iloc[i].values) for i in range(len(ds))],
                'predicted_label': [np.argmax(ds.iloc[i].values) for i in range(len(ds))]
            })
            predict.loc[predict['best_proba'] < threshold, 'predicted_label'] = -1
            return pd.DataFrame(predict['predicted_label'])
    
        predict = predict.map_batches(map_predicted_label, batch_format = 'pandas')
        predict = np.ravel(np.array(predict.to_pandas()))

        return predict
    
    def _label_decode(self, predict):
        print('_label_decode')
        decoded = pd.Series(np.empty(len(predict), dtype=object))
        for label, encoded in self._labels_map.items():
            decoded[predict == encoded] = label

        return np.array(decoded)