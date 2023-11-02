import os
import ray
import warnings
import numpy as np
import pandas as pd

from glob import glob
from shutil import rmtree

# Dimensions reduction
from models.preprocessors.tfidf_transformer import TensorTfIdfTransformer
from data.reduction.rdf_features_selection import TensorRDFFeaturesSelection
from data.reduction.truncated_svd_reduction import TensorTruncatedSVDReduction

# Preprocessing
from ray.data.preprocessors import Chain
from models.encoders.model_label_encoder import ModelLabelEncoder
from models.preprocessors.min_max_scaler import TensorMinMaxScaler
from models.encoders.onesvm_label_encoder import OneClassSVMLabelEncoder

# Training
from ray.air.config import ScalingConfig
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDOneClassSVM, SGDClassifier
from models.sklearn.partial_trainer import SklearnPartialTrainer
from models.sklearn.tensor_predictor import SklearnTensorPredictor

# Tuning
from ray.air.config import RunConfig

# Predicting
from ray.train.batch_predictor import BatchPredictor
from models.sklearn.probability_predictor import SklearnTensorProbaPredictor

# Parent class
from models.models_utils import ModelsUtils

TENSOR_COLUMN_NAME = '__value__'
LABELS_COLUMN_NAME = 'labels'

__author__ = 'Nicolas de Montigny'

__all__ = ['SklearnModel']

# Ignore warnings to have a more comprehensible output on stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

class SklearnModel(ModelsUtils):
    """
    Class used to build, train and predict models using Ray with Scikit-learn backend

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
        self._encoded = []
        # Computes
        self._build()

    def preprocess(self, df):
        print('preprocess')
        if self.classifier == 'onesvm':
            self._encoder = OneClassSVMLabelEncoder(self.taxa)
            self._encoded = np.array([1,-1], dtype = np.int32)
            labels = np.array(['bacteria', 'unknown'], dtype = object)
        else:
            self._encoder = ModelLabelEncoder(self.taxa)
        
        self._preprocessor = Chain(
            TensorTfIdfTransformer(self.kmers),
            TensorRDFFeaturesSelection(self.kmers, self.taxa),
        )
        self._encoder.fit(df)
        df = self._preprocessor.fit_transform(df)
        self.kmers = self._preprocessor.preprocessors[1].stats_['cols_keep']
        self._reductor = TensorTruncatedSVDReduction(self.kmers)
        self._reductor.fit(df)

        # Labels mapping
        if self.classifier != 'onesvm':
            labels = list(self._encoder.stats_[f'unique_values({self.taxa})'].keys())
            self._encoded = np.arange(len(labels))
            labels = np.append(labels, 'unknown')
            self._encoded = np.append(self._encoded, -1)
        self._labels_map = zip(labels, self._encoded)

    def _label_decode(self, predict):
        print('_label_decode')
        decoded = pd.Series(np.empty(len(predict), dtype=object))
        for label, encoded in self._labels_map:
            decoded[predict == encoded] = label

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

        y_true = np.array(y_true)
        y_true = list(y_true)
        
        y_pred = self._predict_cv(df_test.drop_columns([self.taxa]))
        
        self._cv_score(y_true, y_pred)

    def _build(self):
        print('_build')
        if self.classifier == 'onesvm':
            print('Training bacterial extractor with One Class SVM')
            self._clf = SGDOneClassSVM()
            self._train_params = {
                'nu' : 0.026441491,
                'learning_rate' : 'constant',
                'tol' : 1e-3,
                'eta0' : 0.001
            }
        elif self.classifier == 'linearsvm':
            print('Training bacterial / host classifier with SGD')
            self._clf = SGDClassifier()
            self._train_params = {
                'loss' : 'hinge',
                'penalty' : 'elasticnet',
                'alpha' : 141.6146176,
                'learning_rate' : 'adaptive',
                'eta0' : 0.001,
                'n_jobs' : -1
            }
        elif self.classifier == 'sgd':
            print('Training multiclass SGD classifier')
            self._clf = SGDClassifier()
            self._train_params = {
                'alpha' : 173.5667373,
                'learning_rate' : 'optimal',
                'loss': 'modified_huber',
                'penalty' : 'l2'
            }
        elif self.classifier == 'mnb':
            print('Training multiclass Multinomial Naive Bayes classifier')
            self._clf = MultinomialNB()
            self._train_params = {
                'alpha' : 0.243340248,
                'fit_prior' : True
            }

    def _fit_model(self, datasets):
        print('_fit_model')
        for name, ds in datasets.items():
            ds = ds.drop_columns(['id'])
            ds = self._encoder.transform(ds)
            ds = self._preprocessor.transform(ds)
            ds = self._reductor.transform(ds)
            datasets[name] = ray.put(ds)
        try:
            training_labels = self._encoded.copy()
            training_labels = np.delete(
                training_labels, np.where(training_labels == -1))
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

    def _predict_cv(self, df):
        print('_predict_cv')
        if df.count() > 0:
            predict_kwargs = {'features':self.kmers, 'num_estimator_cpus':-1}
            self._predictor = BatchPredictor.from_checkpoint(self._model_ckpt, SklearnTensorPredictor)
            predictions = self._predictor.predict(df, batch_size = self.batch_size, feature_columns = [TENSOR_COLUMN_NAME], **predict_kwargs)
            predictions = np.array(predictions.to_pandas()).reshape(-1)

            return self._label_decode(predictions)
        else:
            raise ValueError('No data to predict')
        
    def predict(self, df, threshold = 0.8):
        print('predict')
        if df.count() > 0:
            df = self._preprocessor.transform(df)
            df = self._reductor.transform(df)
            if self.classifier == 'onesvm':
                predict_kwargs = {'features':self.kmers, 'num_estimator_cpus':-1}
                self._predictor = BatchPredictor.from_checkpoint(self._models_collection['domain'], SklearnTensorPredictor)
                predictions = self._predictor.predict(df, batch_size = self.batch_size, feature_columns = [TENSOR_COLUMN_NAME], **predict_kwargs)
                predictions = np.array(predictions.to_pandas()).reshape(-1)
            else:
                predict_kwargs = {'features':self.kmers, 'num_estimator_cpus':-1}
                self._predictor = BatchPredictor.from_checkpoint(self._model_ckpt, SklearnTensorProbaPredictor)
                predictions = self._predictor.predict(df, batch_size = self.batch_size, feature_columns = [TENSOR_COLUMN_NAME], **predict_kwargs)
                predictions = self._prob_2_cls(predictions, len(self._encoded), threshold)
            return self._label_decode(predictions)    
        else:
            raise ValueError('No data to predict')

    def _prob_2_cls(self, predict, nb_cls, threshold):
        print('_prob_2_cls')
        def map_predicted_label(df : pd.DataFrame):
            predict = pd.DataFrame({
                'best_proba': [max(df.iloc[i].values) for i in range(len(df))],
                'predicted_label': [np.argmax(df.iloc[i].values) for i in range(len(df))]
            })
            predict.loc[predict['best_proba'] < threshold, 'predicted_label'] = -1
            return pd.DataFrame(predict['predicted_label'])

        if nb_cls == 1:
            predict = np.round(abs(np.concatenate(predict.to_pandas()['predictions'])))
        else:
            predict = predict.map_batches(map_predicted_label, batch_format = 'pandas')
            predict = np.ravel(np.array(predict.to_pandas()))

        return predict