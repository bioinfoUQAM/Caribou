import os
import ray
import warnings
import numpy as np
import pandas as pd

# Training
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from ray.train.sklearn import SklearnTrainer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

# Tuning
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.air.config import RunConfig, ScalingConfig

# Predicting
from ray.train.sklearn import SklearnPredictor
from ray.train.batch_predictor import BatchPredictor

# Parent class
from models.ray_utils import ModelsUtils


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
        super().__init__(classifier, outdir_results, batch_size, k, taxa, kmers_list, verbose)
        # Parameters
        if classifier in ['onesvm','linearsvm']:
            self.clf_file = '{}bacteria_binary_classifier_K{}_{}_{}_model.jb'.format(outdir_model, k, classifier, dataset)
        else:
            self.clf_file = '{}{}_multiclass_classifier_K{}_{}_{}_model.jb'.format(outdir_model, taxa, k, classifier, dataset)
        # Computes
        self._build()

    def _training_preprocess(self, X, y):
        print('_training_preprocess')
        print(X.to_pandas())
        print(y)
        sys.exit()
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
        encoded = np.append(np.unique(df.to_pandas()[self.taxa]), -1)
        labels = np.append(labels, 'unknown')
        self._labels_map = zip(labels, encoded)
        return df

    def _build(self):
        print('_build')
        if self.classifier == 'onesvm':
            print('Training bacterial extractor with One Class SVM')
            self._clf = OneClassSVM()
            self._train_params = {
                'kernel' : 'rbf',
                'nu' : 0.05
            }
            self._tuning_params = {
                'params' : {
                    'kernel' : tune.choice(['rbf','poly','sigmoid']),
                    'gamma' : tune.grid_search(np.linspace(0.1,1,10))
                }
            }
        elif self.classifier == 'linearsvm':
            print('Training bacterial / host classifier with Linear SVM')
            self._clf = LinearSVC()
            self._train_params = {
                'penalty' : 'l2'
            }
            self._tuning_params = {
                'params' : {
                    'loss' : tune.choice(['hinge','squared_hinge']),
                    'C' : tune.grid_search(np.logspace(-3,3))
                }
            }
        elif self.classifier == 'sgd':
            print('Training multiclass SGD classifier')
            self._clf = SGDClassifier()
            self._train_params = {
                'loss' : 'squared_error'
            }
            self._tuning_params = {
                'params' : {
                    'loss' : tune.choice(['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
                    'penalty' : tune.choice(['l2', 'l1', 'elasticnet']),
                    'alpha' : tune.grid_search(np.logspace(-4,4)),
                    'learning_rate' : tune.choice(['constant','optimal','invscaling','adaptive']),
                    'eta0' : tune.grid_search(np.logspace(-4,4))
                }
            }
        elif self.classifier == 'svm':
            print('Training multiclass Linear SVM classifier')
            self._clf = SVC()
            self._train_params = {
                'kernel' : 'rbf',
                'probability' : True
            }
            self._tuning_params = {
                'params' : {
                    'kernel' : tune.choice(['rbf','poly','sigmoid']),
                    'C' : tune.grid_search(np.logspace(-3,3)),
                    'gamma' : tune.grid_search(np.linspace(0.1,1,10))
                }
            }
        elif self.classifier == 'mlr':
            print('Training multiclass Multinomial Logistic Regression classifier')
            self._clf = LogisticRegression()
            self._train_params = {
                'solver' : 'saga',
                'multi_class' : 'multinomial'
            }
            self._tuning_params = {
                'params' : {
                    'penalty' : tune.choice(['l1', 'l2', 'elasticnet', 'none']),
                    'C' : tune.grid_search(np.logspace(-3,3)),
                    'l1_ratio' : tune.grid_search(np.linspace(0.1,0.9,10))
                }
            }
        elif self.classifier == 'mnb':
            print('Training multiclass Multinomial Naive Bayes classifier')
            self._clf = MultinomialNB()
            self._train_params = {
                'alpha' : '1.0'
            }
            self._tuning_params = {
                'params' : {
                    'alpha' : tune.grid_search(np.linspace(0,1,10)),
                    'fit_prior' : tune.choice([True, False])
                }
            }

    def _fit_model(self, datasets):
        print('_fit_model')
        # Define trainer
        self._trainer = SklearnTrainer(
            estimator = self._clf,
            label_column = self.taxa,
            params = self._train_params,
            scoring = 'f1_weighted',
            datasets = datasets
        )
        # Define tuner
        self._tuner = Tuner(
            self._trainer,
            param_space = self._tuning_params,
            tune_config = TuneConfig(
                metric = 'test/test_score',
                mode = 'max',
            ),
            scaling_config = ScalingConfig(
                trainer_ressources = self._n_workers
            ),
            run_config = RunConfig(
                name = self.classifier,
                verbose = 1
            )
        )
        # Train / tune execution
        # The Trainable/training function is too large for grpc resource limit. Check that its definition is not implicitly capturing a large array or other object in scope. Tip: use tune.with_parameters() to put large objects in the Ray object store.
        tuning_result = self._tuner.fit()
        self._model_ckpt = tuning_result.get_best_result().checkpoint

    def predict(self, df, threshold = 0.8, cv = False):
        print('predict')
        if not cv:
            df = self._predict_preprocess(df)
        # Define predictor
        self._predictor = BatchPredictor.from_checkpoint(self._model_ckpt, SklearnPredictor)
        # Make predictions
        predictions = self._predictor.predict(df, batch_size = self.batch_size)
        if cv:
            return predictions
        else:
            return self._label_decode(predictions, threshold)
