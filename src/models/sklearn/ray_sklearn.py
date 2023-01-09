import os
import ray
import warnings
import numpy as np
import pandas as pd

from glob import glob
from shutil import rmtree

# Preprocessing
from models.ray_tensor_min_max import TensorMinMaxScaler
from ray.data.preprocessors import BatchMapper, LabelEncoder
from models.sklearn.ray_sklearn_onesvm_encoder import OneClassSVMLabelEncoder

# Training
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDOneClassSVM, SGDClassifier

# Tuning
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.air.config import RunConfig, ScalingConfig

# Predicting
from ray.train.sklearn import SklearnPredictor
from ray.train.batch_predictor import BatchPredictor

# Parent class
from models.ray_utils import ModelsUtils
from models.sklearn.ray_sklearn_partial_trainer import SklearnPartialTrainer
from models.sklearn.ray_sklearn_probability_predictor import SklearnProbaPredictor


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
        self._encoded = []
        # Computes
        self._build()

    def _training_preprocess(self, X, y):
        print('_training_preprocess')
        labels = np.unique(y[self.taxa])
        self._preprocessor = TensorMinMaxScaler(self.kmers)
        self._preprocessor.fit(X)
        self._label_encode(y, labels)
        df = self._zip_X_y(X, y)
        return df

    def _label_encode(self, df, labels):
        print('_label_encode')
        df = ray.data.from_pandas(df)
        if self.classifier == 'onesvm':
            self._encoder = OneClassSVMLabelEncoder(self.taxa)
            self._encoder.fit(df)
            encoded = np.array([1,-1], dtype = np.int32)
            labels = np.array(['bacteria', 'unknown'], dtype = object)
        else:
            self._encoder = LabelEncoder(self.taxa)
            self._encoder.fit(df)
            self._encoded = np.arange(len(labels))
            encoded = np.append(self._encoded, -1)
            labels = np.append(labels, 'unknown')
        self._labels_map = zip(labels, encoded)

    def _label_decode(self, predict):
        print('_label_decode')
        decoded = pd.Series(np.empty(len(predict), dtype=object))
        for label, encoded in self._labels_map:
            decoded[predict == encoded] = label
        return decoded

    def train(self, X, y, kmers_ds, cv = True):
        print('train')

        df = self._training_preprocess(X, y)
        if cv:
            self._cross_validation(df, kmers_ds)
        else:
            df = df.drop_columns(['id'])
            datasets = {'train' : ray.put(df)}
            self._fit_model(datasets)

    def _cross_validation(self, df, kmers_ds):
        print('_cross_validation')
        
        df_train, df_test = df.train_test_split(0.2, shuffle=True)
        df_test = self._sim_4_cv(df_test, kmers_ds, '{}_test'.format(self.dataset))

        df_train = df_train.drop_columns(['id'])
        df_test = df_test.drop_columns(['id'])

        datasets = {'train' : ray.put(df_train)}
        self._fit_model(datasets)

        df_test = self._encoder.transform(df_test)
        y_true = []
        for row in df_test.iter_rows():
            y_true.append(row[self.taxa])

        y_pred = self.predict(df_test.drop_columns([self.taxa]), cv = True)

        for file in glob(os.path.join( os.path.dirname(kmers_ds['profile']), '*sim*')):
            if os.path.isdir(file):
                rmtree(file)
            else:
                os.remove(file)

        self._cv_score(y_true, y_pred)

    def _build(self):
        print('_build')
        if self.classifier == 'onesvm':
            print('Training bacterial extractor with One Class SVM')
            self._clf = SGDOneClassSVM()
            self._train_params = {
                'nu' : 0.1,
                'learning_rate': 'invscaling',
                'eta0' : 1000,
                'tol' : 1e-4
            }
        elif self.classifier == 'linearsvm':
            print('Training bacterial / host classifier with SGD')
            self._clf = SGDClassifier()
            self._train_params = {
                'alpha' : 0.045,
                'eta0' : 1000,
                'learning_rate': 'adaptive',
                'loss' : 'modified_huber',
                'penalty' : 'elasticnet'
            }
        elif self.classifier == 'sgd':
            print('Training multiclass SGD classifier')
            self._clf = SGDClassifier()
            self._train_params = {
                'alpha' : 0.045,
                'learning_rate' : 'optimal',
                'loss': 'log_loss',
                'penalty' : 'elasticnet'
            }
        elif self.classifier == 'mnb':
            print('Training multiclass Multinomial Naive Bayes classifier')
            self._clf = MultinomialNB()
            self._train_params = {
                'alpha' : 1.0,
                'fit_prior' : True
            }

    def _fit_model(self, datasets):
        print('_fit_model')
        for name, ds in datasets.items():
            ds = ray.get(ds)
            ds = self._preprocessor.transform(ds)
            ds = self._encoder.transform(ds)
            datasets[name] = ray.put(ds)

        # Define trainer
        self._trainer = SklearnPartialTrainer(
            estimator = self._clf,
            label_column = self.taxa,
            labels_list = self._encoded,
            params = self._train_params,
            datasets = datasets,
            batch_size = self.batch_size,
            set_estimator_cpus = True,
            scaling_config = ScalingConfig(
                trainer_resources = {
                    'CPU' : int(os.cpu_count()*0.8)
                }
            ),
            run_config = RunConfig(
                name = self.classifier,
                local_dir = self._workdir
            )
        )

        # Training execution
        result = self._trainer.fit()
        self._model_ckpt = result.checkpoint

    def predict(self, df, threshold = 0.8, cv = False):
        print('predict')
        if df.count() > 0:
            df = self._preprocessor.transform(df)
            if self.classifier == 'onesvm':
                self._predictor = BatchPredictor.from_checkpoint(self._model_ckpt, SklearnPredictor)
                predictions = self._predictor.predict(df, batch_size = self.batch_size)
                predictions = np.array(predictions.to_pandas()).reshape(-1)
            else:
                self._predictor = BatchPredictor.from_checkpoint(self._model_ckpt, SklearnProbaPredictor)
                predictions = self._predictor.predict(df, batch_size = self.batch_size)
                predictions = self._prob_2_cls(predictions, len(self._encoded), threshold)
            
            if cv:
                return predictions
            else:
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
            predict.loc[predict['best_proba'] < threshold, 'predicted_label'] = 0
            return pd.DataFrame(predict['predicted_label'])

        if nb_cls == 1:
            predict = np.round(abs(np.concatenate(predict.to_pandas()['predictions'])))
        else:
            mapper = BatchMapper(map_predicted_label, batch_format = 'pandas')
            predict = mapper.transform(predict)
            predict = np.ravel(np.array(predict.to_pandas()))
        
        return predict
        