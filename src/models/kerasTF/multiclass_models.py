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
from models.multiclass_utils import MulticlassUtils

# Training
import tensorflow as tf
from ray.air import session
# from ray.air.integrations.keras import Callback
from ray.air.config import ScalingConfig
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.train.tensorflow import TensorflowTrainer, TensorflowCheckpoint

# Tuning
from ray.air.config import RunConfig

# Predicting
from tensorflow.keras.models import load_model
from ray.train.tensorflow import TensorflowPredictor
from ray.train.batch_predictor import BatchPredictor

# Data
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed

__author__ = 'Nicolas de Montigny'

__all__ = ['KerasTFModel']

TENSOR_COLUMN_NAME = '__value__'
LABELS_COLUMN_NAME = 'labels'

# Ignore warnings to have a more comprehensible output on stdout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

class KerasTFMulticlassModels(KerasTFModels, MulticlassUtils):
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
        self._nb_classes = None

    # Models predicting
    #########################################################################################################

    def _get_abs_pred(self, predictions):
        print('_get_abs_pred')
        return np.argmax(predictions, axis = 1)

    def _get_threshold_pred(self, predictions, threshold):
        print('_get_threshold_pred')
        pred = pd.DataFrame({
            'proba': [np.max(arr) for arr in predictions],
            'label' : [np.argmax(arr) for arr in predictions]
        })
        pred.loc[pred['proba'] < threshold, 'label'] = -1

        return pred['label'].to_numpy(dtype = np.int32)

