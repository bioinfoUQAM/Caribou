
import math

from tensorflow.keras.optimizers import Adam
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

def build_LSTM(kmers_length):
    """
    Function extracted from module seeker/train_model/train_model.py of
    Seeker package [Auslander et al. 2020]
    """
    # Initialize a sequential model
    model = Sequential()

    # Add LSTM layer
    model.add(LSTM(5, input_shape=(kmers_length,)))

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
    model = Sequential(kmers_length)

    # Add LSTM layer
    model.add(Conv2D(5, input_shape=(kmers_length,)))

    # Add Dense NN layer
    model.add(Dense(1, activation='tanh'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def build_deepLSTM(kmers_length):
    """
    Function adapted from module deeplasmid/classifier/dl/DL_Model.py of
    Deeplasmid package [Andreopoulos et al. 2021]
    """

    input = Input(shape=(kmers_length,))

    #input is the kmers encoded sequences and it goes into an LSTM.
    net = LSTM(40, activation='tanh',recurrent_dropout=0.05,dropout=0.1,name='A_%d'%40,return_sequences=True) (input)
    net = LSTM(40, activation='tanh',recurrent_dropout=0.05,dropout=0.1,name='B_%d'%40) (net)

    net = Dense(10*2, activation='relu', name='C_%d'%(10*2))(net)
    net = Dropout(0.1,name='fr_%.1f'%0.1)(net)
    net = Dense(10, activation='relu', name='D_%d'%10)(net)
    net = Dropout(0.1,name='fr_same')(net)
    outputs = Dense(1, activation='tanh', name='score')(net)
    model = Model(inputs=input, outputs=outputs)
    model.compile_model()

    return model
