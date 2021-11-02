
import math

from tensorflow.keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv2D, MaxPooling1D, ReLU

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

def build_LSTM(kmers_length, batch_size):
    """
    Function extracted from module seeker/train_model/train_model.py of
    Seeker package [Auslander et al. 2020] and
    changes inspired by module DeepMicrobes/models/cnn_lstm.py of
    DeepMicrobes package [Liang et al. 2020]
    """

# ADAPT FROM DEEPMICROBES GITHUB
    # Initialize a sequential model
    model = Sequential()

    # Add LSTM layer
    model.add(LSTM(5, input_shape=(batch_size, kmers_length)))

    # Add Dense NN layer
    model.add(Dense(1, activation='tanh'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def build_CNN(kmers_length, k, batch_size):
    """
    Function inspired by module DeepMicrobes/models/cnn_lstm.py of
    DeepMicrobes package [Liang et al. 2020]
    """
    input = Input(shape=(batch_size, kmers_length))

    net = Conv1D(filters = 1024, kernel_size = 30, activation = 'relu') (input)
    net = MaxPooling1D(pool_size = 15, strides = 15) (net)
    length_list, length_max, batch_size = batch_stat(inputs)
# FINISH ADAPTING FROM DEEPMICROBES GITHUB
    net = LSTM(1024, ) (net)
    length_list, batch_size, initializer=tf.contrib.layers.xavier_initializer()
    net = ReLU() (net)
    net = Dense() (net)
    net = Dense() (net)
    outputs = Dense() (net)

    model = Model(inputs=input, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def build_deepLSTM(kmers_length, batch_size):
    """
    Function adapted from module deeplasmid/classifier/dl/DL_Model.py of
    Deeplasmid package [Andreopoulos et al. 2021]
    """

    input = Input(shape=(batch_size, kmers_length))

    #input is the kmers encoded sequences and it goes into an LSTM.
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
