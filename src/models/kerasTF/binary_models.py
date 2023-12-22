import os
import gc
import warnings
import numpy as np
import pandas as pd

# Preprocessing
from ray.data.preprocessors import LabelEncoder, Chain
from models.encoders.model_label_encoder import ModelLabelEncoder
from models.preprocessors.min_max_scaler import TensorMinMaxScaler
from models.encoders.one_hot_tensor_encoder import OneHotTensorEncoder
from models.preprocessors.tfidf_transformer import TensorTfIdfTransformer

# Parent class / models
from models.kerasTF.models import KerasTFModels
from models.kerasTF.build_neural_networks import *

# Training
import tensorflow as tf
from ray.air import session
# from ray.air.integrations.keras import Callback
from ray.air.config import ScalingConfig
from models.kerasTF.models import train_func_CPU, train_func_GPU, build_model
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.train.tensorflow import TensorflowTrainer, TensorflowCheckpoint

# Tuning
from ray.air.config import RunConfig

# Predicting
from ray.train.tensorflow import TensorflowPredictor
from ray.train.batch_predictor import BatchPredictor

__author__ = 'Nicolas de Montigny'

__all__ = ['KerasTFModel']

TENSOR_COLUMN_NAME = '__value__'
LABELS_COLUMN_NAME = 'labels'

# Ignore warnings to have a more comprehensible output on stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

class KerasTFBinaryModels(KerasTFModels):
    """
    Class used to build, train and predict models using Ray with Keras Tensorflow backend

    ----------
    Attributes
    ----------

    clf_file : string
        Path to a file containing the trained model for this object

    nb_classes : int
        Number of classes for learning

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
        self._nb_classes = 2

        if self.classifier == 'attention':
            print('Training bacterial / host classifier based on Attention Weighted Neural Network')
        elif self.classifier == 'lstm':
            print('Training bacterial / host classifier based on Shallow LSTM Neural Network')
        elif self.classifier == 'deeplstm':
            print('Training bacterial / host classifier based on Deep LSTM Neural Network')
        
    # Model predicting
    #########################################################################################################
    
    def _get_abs_pred(self, predictions):
        print('_get_abs_pred')
        return np.round(np.ravel(predictions))
        # predict = pd.DataFrame({
        #     'proba': np.ravel(predictions),
        #     'predicted_label' : np.full(len(predictions), -1)
        # })
        # predict.loc[predict['proba'] > 0.5, 'predicted_label'] = 1
        # predict.loc[predict['proba'] < 0.5, 'predicted_label'] = 0

        # return predict

    def _get_threshold_pred(self, predictions, threshold):
        print('_get_threshold_pred')
        lower_threshold = 0.5 - (threshold * 0.5)
        upper_threshold = 0.5 + (threshold * 0.5)
        
        predict = pd.DataFrame({
            'proba': np.ravel(predictions),
            'label' : np.full(len(predictions), -1)
        })

        predict.loc[predict['proba'] >= upper_threshold, 'label'] = 1
        predict.loc[predict['proba'] <= lower_threshold, 'label'] = 0
        
        return predict['label'].to_numpy(dtype = np.int32)