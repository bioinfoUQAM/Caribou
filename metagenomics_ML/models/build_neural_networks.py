
import math

from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv2D

from models.attentionLayer import AttentionWeightedAverage

__author__ = "nicolas"

def build_gradl():
# MUST BUILD NEURAL NETWORK FROM READING ARTICLE TO IMPLEMENT
    print("Possibility")

def build_attention(kmers_length):
    """
    Function extracted from module virnet/NNClassifier.py of
    VirNet package [Abdelkareem et al. 2018]
    """
    inp = Input(shape = (int(math.ceil(kmers_length * 1.0 / 5)),) )
    x = Embedding(4, 128)(inp)
    for i in range(2):
        x = LSTM(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1 )(x)
    x = AttentionWeightedAverage()(x)
    x = Dense(128, activation = "relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation = "sigmoid")(x)

    model = Model(inputs = inp, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.001), metrics = ['accuracy'])

    return model

def build_LSTM():
    """
    Function extracted from module seeker/train_model/train_model.py of
    Seeker package [Auslander et al. 2020]
    """
    # Initialize a sequential model
    model = Sequential()

    # Add LSTM layer
    model.add(LSTM(5, input_shape=(NUC_COUNT, 1000)))

    # Add Dense NN layer
    model.add(Dense(1, activation='tanh'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def build_CNN():
    """
    Function extracted from module seeker/train_model/train_model.py of
    Seeker package [Auslander et al. 2020]
    """
    # Initialize a sequential model
    model = Sequential()

    # Add LSTM layer
    model.add(Conv2D(5, input_shape=(NUC_COUNT, 1000)))

    # Add Dense NN layer
    model.add(Dense(1, activation='tanh'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
