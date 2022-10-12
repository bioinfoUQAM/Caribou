import os
import ray
import warnings
import numpy as np
import pandas as pd

from glob import glob
from shutil import rmtree
# Preprocessing
from ray.data.preprocessors import MinMaxScaler, LabelEncoder, Chain, SimpleImputer

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
from models.ray_sklearn_partial_trainer import SklearnPartialTrainer


__author__ = 'Nicolas de Montigny'

__all__ = ['SklearnModel']

# Ignore warnings to have a more comprehensible output on stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

class SklearnModel(ModelsUtils):
    # https://docs.ray.io/en/master/ray-air/examples/sklearn_example.html
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
    def __init__(self, classifier, dataset, outdir_model, outdir_results, batch_size, k, taxa, kmers_list, verbose):
        super().__init__(classifier, dataset, outdir_model, outdir_results, batch_size, k, taxa, kmers_list, verbose)
        # Parameters
        self._encoded = []
        # if classifier in ['onesvm','linearsvm']:
        #     self.clf_file = '{}bacteria_binary_classifier_K{}_{}_{}_model.jb'.format(outdir_model, k, classifier, dataset)
        # else:
        #     self.clf_file = '{}{}_multiclass_classifier_K{}_{}_{}_model.jb'.format(outdir_model, taxa, k, classifier, dataset)
        # Computes
        self._build()

    def _training_preprocess(self, X, y):
        print('_training_preprocess')
        df = X.add_column([self.taxa, 'id'], lambda x: y)
        self._preprocessor = Chain(
            SimpleImputer(
                self.kmers,
                strategy='constant',
                fill_value=0),
            MinMaxScaler(self.kmers)
        )
        df = self._preprocessor.fit_transform(df)
        labels = np.unique(y[self.taxa])
        df = self._label_encode(df, labels)
        return df

    def _label_encode(self, df, labels):
        print('_label_encode')
        self._encoder = LabelEncoder(self.taxa)
        df = self._encoder.fit_transform(df)
        self._encoded = np.unique(df.to_pandas()[self.taxa])
        encoded = np.append(self._encoded, -1)
        labels = np.append(labels, 'unknown')
        self._labels_map = zip(labels, encoded)
        return df

    def _label_decode(self, predict):
        print('_label_decode')
        predict = np.array(predict.to_pandas())
        decoded = pd.Series(np.empty(len(predict), dtype=object))
        for label, encoded in self._labels_map:
            decoded[predict == encoded] = label
        return decoded

    def _cross_validation(self, df, kmers_ds):
        print('_cross_validation')

        df_train, df_test = df.train_test_split(0.2, shuffle = True)
        df_train, df_val = df_train.train_test_split(0.2, shuffle = True)

        df_train = df_train.drop_columns(['id'])

        df_val = self._sim_4_cv(df_val, kmers_ds, '{}_val'.format(self.dataset))
        df_test = self._sim_4_cv(df_test, kmers_ds, '{}_test'.format(self.dataset))

        datasets = {'train' : ray.put(df_train), 'validation' : ray.put(df_val)}
        self._fit_model(datasets)

        y_true = df_test.to_pandas()[self.taxa]
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
                'nu' : 0.05,
                'tol' : 1e-4
            }
        elif self.classifier == 'linearsvm':
            print('Training bacterial / host classifier with SGD')
            self._clf = SGDClassifier()
            self._train_params = {
                'loss' : 'squared_error'
            }
        elif self.classifier == 'sgd':
            print('Training multiclass SGD classifier')
            self._clf = SGDClassifier()
            self._train_params = {
                'loss' : 'squared_error'
            }
        elif self.classifier == 'mnb':
            print('Training multiclass Multinomial Naive Bayes classifier')
            self._clf = MultinomialNB()
            self._train_params = {
                'alpha' : '1.0'
            }

    def _fit_model(self, datasets):
        print('_fit_model')
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
                    'CPU' : 5
                }
            ),
            run_config = RunConfig(
                name = self.classifier,
                local_dir = self._workdir
            )
        )

        # Training execution
        result = trainer.fit()
        self._model_ckpt = result.checkpoint

    def predict(self, df, threshold = 0.8, cv = False):
        print('predict')
        if not cv:
            df = self._predict_preprocess(df)
        # Define predictor
        self._predictor = BatchPredictor.from_checkpoint(self._model_ckpt, SklearnPredictor)
        # Make predictions
        predictions = self._predictor.predict(df, batch_size = self.batch_size)

        return self._label_decode(predictions, threshold)
