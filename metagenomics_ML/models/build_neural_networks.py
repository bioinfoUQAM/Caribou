
import math

from tensorflow import cast
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotNormal
from keras.models import Model, Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv2D, MaxPooling1D, ReLU, Concatenate, Flatten

from models.attentionLayer import AttentionWeightedAverage

__author__ = "nicolas"

# Host extraction
def build_attention(kmers_length):
    """
    Function extracted from module virnet/NNClassifier.py of
    VirNet package [Abdelkareem et al. 2018]
    """

    inputs = Input(shape = (kmers_length,))
    x = Embedding(4, 128)(inputs)
    x = LSTM(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1 )(x)
    x = LSTM(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1 )(x)
    x = AttentionWeightedAverage()(x)
    x = Dense(128, activation = "relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation = "sigmoid")(x)

    model = Model(inputs = inputs, outputs = x)
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

    inputs = Input(shape=(batch_size, kmers_length))

    net = LSTM(40, activation='tanh',recurrent_dropout=0.05,dropout=0.1,name='A_%d'%40,return_sequences=True) (inputs)
    net = LSTM(40, activation='tanh',recurrent_dropout=0.05,dropout=0.1,name='B_%d'%40) (net)

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
    Function adapted in keras from module DeepMicrobes/models/embed_lstm_attention.py of
    DeepMicrobes package [Liang et al. 2020]
    """
    inputs = Input(shape = (batch_size, kmers_length))
    net = Embedding(8390658, 100, kernel_initializer = 'glorot_normal')(inputs)
    net = LSTM((300, cast(kmers_length, tf.int32)), kernel_initializer = 'glorot_normal')(net)
    net = Attention(kernel_initializer = 'glorot_normal')(net)
    net = Dense(18000, activation = 'relu', kernel_initializer = 'glorot_normal')(net)
    net = Dropout(0.2, kernel_initializer = 'glorot_normal')(net)
    net = Dense(3000, activation = 'relu', kernel_initializer = 'glorot_normal')(net)
    net = Dropout(0.2, kernel_initializer = 'glorot_normal')(net)
    outputs = Activation('softmax', kernel_initializer = 'glorot_normal')(net)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def build_CNN(nb_classes, nb_kmers):
    """
    Function extracted from module MetagenomicDC/models/CNN.py of
    MetagenomicDC package [Fiannaca et al. 2018]
    """
    model = Sequential()
    model.add(Convolution1D(5,5, border_mode='valid', input_dim = 1, input_length = nb_kmers)) #input_dim
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=2,border_mode='valid'))
    model.add(Convolution1D(10, 5,border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=2,border_mode='valid'))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def build_deepCNN(kmers_length, batch_size):
    """
    Function adapted in keras from module CHEER/Classifier/model/Wcnn.py of
    CHEER package [Shang et al. 2021]
    """
    inputs = Input(shape = (batch_size, kmers_length))
    embed = Embedding(248, 100)(inputs)
    conv1 = Conv2D(256, (3, kmers_length), input_shape = 1, activation = 'relu')(embed)
    conv1 = MaxPooling1D(stride = kmers_length)(conv1)
    conv2 = Conv2D(256, (7, kmers_length), input_shape = 1, activation = 'relu')(embed)
    conv2 = MaxPooling1D(stride = kmers_length)(conv2)
    conv3 = Conv2D(256, (11, kmers_length), input_shape = 1, activation = 'relu')(embed)
    conv3 = MaxPooling1D(stride = kmers_length)(conv3)
    conv4 = Conv2D(256, (15, kmers_length), input_shape = 1, activation = 'relu')(embed)
    conv4 = MaxPooling1D(stride = kmers_length)(conv4)
    net = Concatenate([conv1,conv2,conv3,conv4], axis = 1)
    net = Flatten()(net)
    net = Dense(1024)(net)
    net = Activation("relu")(net)
    net = Dropout(0.5)(net)
    net = Flatten()(net)
    net = Dense(512)(net)
    outputs = Activation('softmax')(net)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
