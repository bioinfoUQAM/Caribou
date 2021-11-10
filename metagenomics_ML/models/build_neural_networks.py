
import math

import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv2D, MaxPooling1D, ReLU

from models.attentionLayer import AttentionWeightedAverage

__author__ = "nicolas"

# Host extraction
def build_attention(kmers_length):
    """
    Function extracted from module virnet/NNClassifier.py of
    VirNet package [Abdelkareem et al. 2018]
    """

    inp = Input(shape = (kmers_length,))
    x = Embedding(4, 128)(inp)
    x = LSTM(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1 )(x)
    x = LSTM(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1 )(x)
    x = AttentionWeightedAverage()(x)
    x = Dense(128, activation = "relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation = "sigmoid")(x)

    model = Model(inputs = inp, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.001), metrics = ['accuracy'])

    return model

# Host extraction
def build_LSTM(kmers_length, batch_size):
    """
    Function extracted from module seeker/train_model/train_model.py of
    Seeker package [Auslander et al. 2020]
    """

    # Initialize a sequential model
    model = Sequential()
    # Add LSTM layer
    model.add(LSTM(5, input_shape=(batch_size, kmers_length)))
    # Add Dense NN layer
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Binary classifier
def build_deepLSTM(kmers_length, batch_size):
    """
    Function adapted from module deeplasmid/classifier/dl/DL_Model.py of
    Deeplasmid package [Andreopoulos et al. 2021]
    """

    input = Input(shape=(batch_size, kmers_length))

    net = LSTM(40, activation='tanh',recurrent_dropout=0.05,dropout=0.1,name='A_%d'%40,return_sequences=True) (input)
    net = LSTM(40, activation='tanh',recurrent_dropout=0.05,dropout=0.1,name='B_%d'%40) (net)

    net = Dense(10*2, activation='relu', name='C_%d'%(10*2))(net)
    net = Dropout(0.1,name='fr_%.1f'%0.1)(net)
    net = Dense(10, activation='relu', name='D_%d'%10)(net)
    net = Dropout(0.1,name='fr_same')(net)
    outputs = Dense(1, activation='tanh', name='score')(net)
    model = Model(inputs=input, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Bacteria classification
# In their article this is the best model built to classify bacterias
def build_LSTM_attention():
    """
    Function extracted from module DeepMicrobes/models/embed_lstm_attention.py of
    DeepMicrobes package [Liang et al. 2020]
    """
    print("To do")

# MAYBEE NOT THIS ONE IF USE WDCNN (COMPARE PERFORMANCES)
def build_CNN():
    """
    Function extracted from module MetagenomicDC/models/CNN.py of
    MetagenomicDC package [Fiannaca et al. 2018]
    """
    print("To do")

def build_DBN():
    """
    Function extracted from module MetagenomicDC/models/DBN.py of
    MetagenomicDC package [Fiannaca et al. 2018]
    """
    print("To do")

def build_WDcnn():
    """
    Function extracted from module CHEER/models/WDcnn.py of
    CHEER package [Shang et al. 2021]
    """
    print("To do")
