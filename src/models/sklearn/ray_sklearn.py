import os
import ray
import warnings
import numpy as np
import pandas as pd

from glob import glob
from shutil import rmtree

# Preprocessing
from models.ray_tensor_min_max import TensorMinMaxScaler
from ray.data.preprocessors import Chain, BatchMapper, LabelEncoder
from models.sklearn.ray_sklearn_onesvm_encoder import OneClassSVMLabelEncoder

# Training
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDOneClassSVM, SGDClassifier

# Tuning
from ray.air.config import RunConfig, ScalingConfig

# Predicting
from ray.train.sklearn import SklearnPredictor
from ray.train.batch_predictor import BatchPredictor
from joblib import Parallel, delayed, parallel_backend

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
        self._encoder = None
        # Computes
        self._build()

    def preprocess(self, df):
        print('preprocess')
        if self.classifier == 'onesvm':
            self._encoder = OneClassSVMLabelEncoder(self.taxa)
            self._encoded = np.array([1,-1], dtype = np.int32)
            labels = np.array(['bacteria', 'unknown'], dtype = object)
        else:
            self._encoder = LabelEncoder(self.taxa)
        
        self._preprocessor = Chain(
            TensorMinMaxScaler(self.kmers),
            self._encoder,
        )
        self._preprocessor.fit(df)
        if self.classifier != 'onesvm':
            labels = list(self._preprocessor.preprocessors[1].stats_[f'unique_values({self.taxa})'].keys())
            self._encoded = np.arange(len(labels))
            labels = np.append(labels, 'unknown')
            self._encoded = np.append(self._encoded, -1)
        self._labels_map = zip(labels, self._encoded)

    def _label_decode(self, predict):
        print('_label_decode')
        decoded = pd.Series(np.empty(len(predict), dtype=object))
        for label, encoded in self._labels_map:
            decoded[predict == encoded] = label

        return decoded

    def train(self, datasets, kmers_ds, cv = True):
        print('train')
        
        if cv:
            self._cross_validation(datasets, kmers_ds)
        else:
            if self.classifier in ['onesvm','linearsvm']:
                self._fit_model_binary(datasets)
            else:
                self._fit_model_multiclass(datasets)

    def _cross_validation(self, datasets, kmers_ds):
        print('_cross_validation')
        
        df_test = datasets.pop('test')

        if self.classifier in ['onesvm', 'linearsvm']:
            self._fit_model_binary(datasets)
        else:
            self._fit_model_multiclass(datasets)


        df_test = self._preprocessor.preprocessors[0].transform(df_test)

        y_true = []
        for row in df_test.iter_rows():
            y_true.append(row[self.taxa])

        y_true = np.array(y_true)
        y_true = list(y_true)
        
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

    def _fit_model_binary(self, datasets):
        print('_fit_model_binary')
        for name, ds in datasets.items():
            print(f'dataset preprocessing : {name}')
            ds = ds.drop_columns(['id'])
            ds = self._preprocessor.transform(ds)
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
            label_column=self.taxa,
            labels_list=training_labels,
            features_list=self.kmers,
            params=self._train_params,
            datasets=datasets,
            batch_size=self.batch_size,
            set_estimator_cpus=True,
            scaling_config=ScalingConfig(
                trainer_resources={
                    'CPU': int(os.cpu_count()*0.8)
                }
            ),
            run_config=RunConfig(
                name=self.classifier,
                local_dir=self._workdir
            ),
        )

        # Training execution
        result = self._trainer.fit()
        self._models_collection['domain'] = result.checkpoint

    def _fit_model_multiclass(self, datasets):
        print('_fit_model_multiclass')
        training_collection = datasets.pop('train')
        for name, ds in datasets.items():
            print(f'dataset preprocessing : {name}')
            ds = ds.drop_columns(['id'])
            ds = self._preprocessor.transform(ds)
            datasets[name] = ray.put(ds)

        try:
            training_labels = self._encoded.copy()
            training_labels = np.delete(training_labels, np.where(training_labels == -1))
        except:
            pass
        
        for tax, ds in training_collection.items():
            ds = ds.drop_columns(['id'])
            ds = self._preprocessor.transform(ds)
            training_ds = {**{'train' : ray.put(ds)}, **datasets}

            # Define trainer
            self._trainer = SklearnPartialTrainer(
                estimator = self._clf,
                label_column = self.taxa,
                labels_list = training_labels,
                features_list = self.kmers,
                params = self._train_params,
                datasets = training_ds,
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
                ),
            )

            # Training execution
            training_result = self._trainer.fit()
            self._models_collection[tax] = training_result.checkpoint

    def predict(self, df, threshold = 0.8, cv = False):
        print('predict')
        if df.count() > 0:
            df = self._preprocessor.preprocessors[0].transform(df)
            if self.classifier == 'onesvm':
                self._predictor = BatchPredictor.from_checkpoint(self._models_collection['domain'], SklearnPredictor)
                predictions = self._predictor.predict(df, batch_size = self.batch_size)
                predictions = np.array(predictions.to_pandas()).reshape(-1)
            elif self.classifier == 'linearsvm':
                self._predictor = BatchPredictor.from_checkpoint(self._models_collection['domain'], SklearnProbaPredictor)
                predictions = self._predictor.predict(df, batch_size = self.batch_size)
                predictions = self._prob_2_cls_binary(predictions, threshold)
            else:
                pred_dct = {}
                for tax, ckpt in self._models_collection.items():
                    self._predictor = BatchPredictor.from_checkpoint(ckpt, SklearnProbaPredictor)
                    pred_dct[tax] = self._predictor.predict(df, batch_size = self.batch_size)
                predictions = self._prob_2_cls_multiclass(pred_dct, df.count(), threshold)

            return self._label_decode(predictions)    
        else:
            raise ValueError('No data to predict')

    def _prob_2_cls_binary(self, predict, threshold):
        print('_prob_2_cls')
        def map_predicted_label(df : pd.DataFrame):
            predict = pd.DataFrame({
                'best_proba': [max(df.iloc[i].values) for i in range(len(df))],
                'predicted_label': [np.argmax(df.iloc[i].values) for i in range(len(df))]
            })
            predict.loc[predict['best_proba'] < threshold, 'predicted_label'] = -1
            return pd.DataFrame(predict['predicted_label'])

        mapper = BatchMapper(map_predicted_label, batch_format = 'pandas')
        predict = mapper.transform(predict)
        predict = np.ravel(np.array(predict.to_pandas()))
    
        return predict
        
    def _prob_2_cls_multiclass(self, pred_dct, nb_records, threshold):
        print('_prob_2_cls')
        def map_predicted_label(df):
            predict = pd.DataFrame({
                'best_proba': [max(df.iloc[i].values) for i in range(len(df))],
                'predicted_label': [np.argmax(df.iloc[i].values) for i in range(len(df))]
            })
            return predict

        global_predict = pd.DataFrame({
            'predict_proba': np.zeros(nb_records, dtype=np.float32),
            'predict_cls': np.zeros(nb_records, dtype=np.int32),
        })
        for tax, local_predict in pred_dct.items():
            with parallel_backend('threading'):
                local_predict = Parallel(n_jobs=-1, prefer='threads', verbose=1)(
                    delayed(map_predicted_label)(batch) for batch in local_predict.iter_batches(batch_size=self.batch_size))
            local_predict = pd.concat(local_predict, ignore_index = True)
            global_predict.loc[global_predict['predict_proba'] < local_predict['best_proba'],'predict_cls'] = np.array(local_predict.loc[local_predict['best_proba'] > global_predict['predict_proba'], 'predicted_label'])
            global_predict.loc[global_predict['predict_proba'] < local_predict['best_proba'],'predict_proba'] = np.array(local_predict.loc[local_predict['best_proba'] > global_predict['predict_proba'], 'best_proba'])
        global_predict.loc[global_predict['predict_proba'] < threshold, 'predict_cls'] = -1
    
        return np.array(global_predict['predict_cls'])