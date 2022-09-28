import os
import ray
import warnings
import numpy as np
import pandas as pd

# Preprocessing
from ray.data.preprocessors import MinMaxScaler, LabelEncoder, Chain, SimpleImputer

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
        super().__init__(classifier, dataset, outdir_results, batch_size, k, taxa, kmers_list, verbose)
        # Parameters
        if classifier in ['onesvm','linearsvm']:
            self.clf_file = '{}bacteria_binary_classifier_K{}_{}_{}_model.jb'.format(outdir_model, k, classifier, dataset)
        else:
            self.clf_file = '{}{}_multiclass_classifier_K{}_{}_{}_model.jb'.format(outdir_model, taxa, k, classifier, dataset)
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
        encoded = np.append(np.unique(df.to_pandas()[self.taxa]), -1)
        labels = np.append(labels, 'unknown')
        self._labels_map = zip(labels, encoded)
        return df

    def _cross_validation(self, df, kmers_ds):
        print('_cross_validation')

        df_train, df_test = df.train_test_split(0.2, shuffle = True)
        df_train, df_val = df_train.train_test_split(0.2, shuffle = True)

        df_train = df_train.drop_columns(['id'])

        df_val = self._sim_4_cv(df_val, kmers_ds, '{}_val'.format(self.dataset))
        df_test = self._sim_4_cv(df_test, kmers_ds, '{}_test'.format(self.dataset))

        datasets = {'train' : df_train, 'validation' : df_val}
        self._fit_model(datasets)

        y_true = df_test.to_pandas()[self.taxa]
        y_pred = self.predict(df_test.drop_columns(['id',self.taxa]), cv = True)

        rmtree(sim_data['profile'])
        for file in glob(os.path.join(sim_outdir, '*sim*')):
            os.remove(file)

        self._cv_score(y_true, y_pred)

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
            datasets = datasets,
            set_estimator_cpus = True,
            scaling_config = ScalingConfig(
                trainer_resources = {
                    'CPU' : self._n_workers
                }
            )
        )

        result = self._trainer.fit()
        print(result)
        sys.exit()

        self._trainer = ray.put(self._trainer)

"""        _fit_model
        Traceback (most recent call last):
          File "/usr/local/bin/Bacteria_extraction_train_cv.py", line 141, in <module>
            bacteria_extraction_train_cv(opt)
          File "/usr/local/bin/Bacteria_extraction_train_cv.py", line 97, in bacteria_extraction_train_cv
            bacteria_extraction(None,
          File "/usr/local/lib/python3.8/dist-packages/models/extraction.py", line 77, in bacteria_extraction
            model.train(X_train, y_train, database_k_mers, cv)
          File "/usr/local/lib/python3.8/dist-packages/models/ray_utils.py", line 125, in train
            self._fit_model(datasets)
          File "/usr/local/lib/python3.8/dist-packages/models/ray_sklearn.py", line 223, in _fit_model
            self._tuner = Tuner(
          File "/usr/local/lib/python3.8/dist-packages/ray/tune/tuner.py", line 144, in __init__
            self._local_tuner = TunerInternal(**kwargs)
          File "/usr/local/lib/python3.8/dist-packages/ray/tune/impl/tuner_internal.py", line 103, in __init__
            self._experiment_checkpoint_dir = self._setup_create_experiment_checkpoint_dir(
          File "/usr/local/lib/python3.8/dist-packages/ray/tune/impl/tuner_internal.py", line 255, in _setup_create_experiment_checkpoint_dir
            path = Experiment.get_experiment_checkpoint_dir(
          File "/usr/local/lib/python3.8/dist-packages/ray/tune/experiment/experiment.py", line 389, in get_experiment_checkpoint_dir
            run_identifier = cls.get_trainable_name(run_obj)
          File "/usr/local/lib/python3.8/dist-packages/ray/tune/experiment/experiment.py", line 330, in get_trainable_name
            raise TuneError("Improper 'run' - not string nor trainable.")
        ray.tune.error.TuneError: Improper 'run' - not string nor trainable."""

        # Define tuner
        self._tuner = Tuner(
            self._trainer,
            param_space = self._tuning_params,
            tune_config = TuneConfig(
                metric = 'validation/test_score',
                mode = 'max',
            ),
            run_config = RunConfig(
                name = self.classifier,
                verbose = 1
            )
        )
        # Train / tune execution
        tuning_result = self._tuner.fit()
        df_tuning = tuning_result.get_dataframe()
        df_tuning.write_csv(os.path.join(self.outdir_results,'tuning_result_{}.csv'.format(self.classifier)))
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
