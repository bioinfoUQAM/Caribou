
import math

import tensorflow as tf
from tensorflow import cast
from tensorflow.keras.initializers import GlorotNormal
from keras.models import Model, Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, ReLU, Concatenate, Flatten, Attention, Activation, Bidirectional

from Caribou.models.attentionLayer import AttentionWeightedAverage

__author__ = "Nicolas de Montigny"

__all__ = ['build_attention','build_LSTM','build_deepLSTM','build_LSTM_attention','build_CNN','build_wideCNN']

# Host extraction
def build_attention(kmers_length):
    """
    Function extracted from module virnet/NNClassifier.py of
    VirNet package [Abdelkareem et al. 2018]
    """

    inputs = Input(shape = (kmers_length,))
    x = Embedding(kmers_length, 128)(inputs)

    x = LSTM(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1 )(x)
    x = LSTM(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1 )(x)
    x = AttentionWeightedAverage()(x)

    x = Dense(128, activation = "relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation = "sigmoid")(x)

    model = Model(inputs = inputs, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model

# Host extraction
def build_LSTM(kmers_length, batch_size):
    """
    Function extracted from module seeker/train_model/train_model.py of
    Seeker package [Auslander et al. 2020]
    """

    model = Sequential()
    model.add(LSTM(5, input_shape=(batch_size, kmers_length)))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Binary classifier
def build_deepLSTM(kmers_length, batch_size):
    """
    Function adapted from module deeplasmid/classifier/dl/DL_Model.py of
    Deeplasmid package [Andreopoulos et al. 2021]
    """

    inputA = Input(shape=(batch_size, kmers_length))
    inputB = Input(shape=(batch_size, 1))

    netA = LSTM(40, activation='tanh',recurrent_dropout=0.05,dropout=0.1,name='A_%d'%40,return_sequences=True) (inputA)
    netA = LSTM(40, activation='tanh',recurrent_dropout=0.05,dropout=0.1,name='B_%d'%40) (netA)

    netB = Dense(100, activation='tanh',recurrent_dropout=0.05,dropout=0.1,name='A_%d'%40,return_sequences=True) (inputB)
    netB = Dense(100, activation='tanh',recurrent_dropout=0.05,dropout=0.1,name='B_%d'%40) (netB)

    net = Concatenate(axis = 1)([netA,netB])

    net = Dense(10*2, activation='relu', name='C_%d'%(10*2))(net)
    net = Dropout(0.1,name='fr_%.1f'%0.1)(net)

    net = Dense(10, activation='relu', name='D_%d'%10)(net)
    net = Dropout(0.1,name='fr_same')(net)

    outputs = Dense(1, activation='tanh', name='score')(net)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def build_LSTM_attention(kmers_length, nb_classes, batch_size):
    """
    Function adapted in keras from module DeepMicrobes/models/embed_lstm_attention.py and
    default values for layers in script DeepMicrobes/models/define_flags.py of
    DeepMicrobes package [Liang et al. 2020]
    """

    inputs = Input(shape = (kmers_length,))
    net = Embedding(kmers_length, 100)(inputs)

    net = Bidirectional(LSTM(300, return_sequences=True))(net)
    net = Attention()([net,net])

    net = Dense((batch_size * 300 * 2), activation = 'relu')(net)
    net = Dropout(0.2)(net)

    net = Dense(nb_classes, activation = 'relu')(net)
    net = Dropout(0.2)(net)

    outputs = Activation('softmax')(net)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def build_CNN(kmers_length, batch_size, nb_classes):
    """
    Function extracted from module MetagenomicDC/models/CNN.py of
    MetagenomicDC package [Fiannaca et al. 2018]
    """

    model = Sequential()

    model.add(Conv1D(5,5, input_shape = (batch_size, kmers_length))) #input_dim
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size = 2))

    model.add(Conv1D(10, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size = 2))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def build_wideCNN(kmers_length, batch_size, nb_classes):
    """
    Function adapted in keras from module CHEER/Classifier/model/Wcnn.py of
    CHEER package [Shang et al. 2021]
    """

    inputs = Input(shape = (batch_size, kmers_length))
    embed = Embedding(248, 100)(inputs)

    conv1 = Conv2D(256, (3, kmers_length), activation = 'relu')(embed)
    conv1 = MaxPooling2D(pool_size = (1,1), strides = kmers_length)(conv1)

    conv2 = Conv2D(256, (7, kmers_length), activation = 'relu')(embed)
    conv2 = MaxPooling2D(pool_size = (1,1), strides = kmers_length)(conv2)

    conv3 = Conv2D(256, (11, kmers_length), activation = 'relu')(embed)
    conv3 = MaxPooling2D(pool_size = (1,1), strides = kmers_length)(conv3)

    conv4 = Conv2D(256, (15, kmers_length), activation = 'relu')(embed)
    conv4 = MaxPooling2D(pool_size = (1,1), strides = kmers_length)(conv4)

    net = Concatenate(axis = 1)([conv1,conv2,conv3,conv4])
    net = Flatten()(net)

    net = Dense(1024)(net)
    net = Activation("relu")(net)
    net = Dropout(0.5)(net)
    net = Flatten()(net)

    net = Dense(nb_classes)(net)
    outputs = Activation('softmax')(net)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
