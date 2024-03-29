import tensorflow as tf

from keras.models import Model, Sequential
from tensorflow.keras import mixed_precision
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Concatenate, Flatten, Attention, Activation, Bidirectional, Reshape, AveragePooling1D

from tensorflow.keras import mixed_precision
from models.kerasTF.attentionLayer import AttentionWeightedAverage

if len(tf.config.list_physical_devices('GPU')) > 0:
    mixed_precision.set_global_policy('mixed_float16')
else:
    mixed_precision.set_global_policy('mixed_bfloat16')

__author__ = "Nicolas de Montigny"

__all__ = ['build_attention','build_LSTM','build_deepLSTM','build_LSTM_attention','build_CNN','build_wideCNN']

# Self-aware binary classifier
def build_attention(nb_features):
    """
    Function extracted from module virnet/NNClassifier.py of
    VirNet package [Abdelkareem et al. 2018]
    https://github.com/alyosama/virnet/blob/master/NNClassifier.py
    """
    inputs = Input(shape = (nb_features,1))
    # x = Embedding(nb_features, 128)(inputs)

    x = LSTM(128, return_sequences = True, dropout = 0.1)(inputs)
    x = LSTM(128, return_sequences = True, dropout = 0.1)(x)
    x = AttentionWeightedAverage()(x)

    x = Dense(128, activation = "relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1)(x)
    x = Activation(activation = "sigmoid", dtype = 'float32')(x)

    model = Model(inputs = inputs, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'], jit_compile = True)

    return model

# Recurrent binary classifier
def build_LSTM(nb_features):
    """
    Function extracted from module seeker/train_model/train_model.py of
    Seeker package [Auslander et al. 2020]
    https://github.com/gussow/seeker/blob/master/train_model/train_model.py
    """
    
    inputs = Input(shape = (nb_features,1))
    # x = Embedding(nb_features, 128)(inputs)

    x = LSTM(128, dropout = 0.1)(inputs)

    x = Dense(1)(x)
    x = Activation(activation = "tanh", dtype = 'float32')(x)

    model = Model(inputs = inputs, outputs = x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], jit_compile = True)

    return model

# Deep recurrent binary classifier
def build_deepLSTM(nb_features):
    """
    Function adapted from module deeplasmid/classifier/dl/DL_Model.py of
    Deeplasmid package [Andreopoulos et al. 2021]
    https://github.com/wandreopoulos/deeplasmid/blob/docker/classifier/dl/DL_Model.py
    """

    inputs = Input(shape=(nb_features,1))

    # netA = Embedding(nb_features, 128)(inputs)
    netA = LSTM(40, activation='tanh',dropout=0.1,name='A_%d'%40,return_sequences=True) (inputs)
    netA = LSTM(40, activation='tanh',dropout=0.1,name='B_%d'%40) (netA)

    netB = Dense(100, activation='tanh',name='G_%d'%40) (inputs)
    netB = Dense(100, activation='tanh',name='H_%d'%40) (netB)
    netB = AveragePooling1D(100) (netB)
    netB = Flatten() (netB)

    net = Concatenate()([netA,netB])

    net = Dense(200, activation='relu', name='C_%d'%(10*2))(net)
    net = Dropout(0.1,name='fr_%.1f'%0.1)(net)

    net = Dense(10, activation='relu', name='D_%d'%10)(net)
    net = Dropout(0.1,name='fr_same')(net)

    net = Dense(1)(net)
    outputs = Activation(activation = "sigmoid", dtype = 'float32')(net)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], jit_compile = True)

    return model

# Recurrent self-aware multiclass classifier
def build_LSTM_attention(nb_features, nb_classes):
    """
    Function adapted in keras from module DeepMicrobes/models/embed_lstm_attention.py and
    default values for layers in script DeepMicrobes/models/define_flags.py of
    DeepMicrobes package [Liang et al. 2020]
    https://github.com/MicrobeLab/DeepMicrobes/blob/master/models/embed_lstm_attention.py
    """

    inputs = Input(shape = (nb_features,1))
    # net = Embedding(nb_features, 100)(inputs)
    net = Bidirectional(LSTM(300, return_sequences=True))(inputs)
    net = Attention(dropout = 0.2)([net,net])
    # MLP
    net = Dense((nb_features * 300 * 2), activation = 'relu')(net)
    net = Dropout(0.2)(net)
    net = Dense(nb_classes, activation = 'relu')(net)
    net = Dropout(0.2)(net)
    net = Flatten()(net)
    net = Dense(nb_classes)(net)
    outputs = Activation('softmax', dtype = 'float32')(net)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'], jit_compile = True)

    return model

# Convolutional multiclass classifier
def build_CNN(nb_features, nb_classes):
    """
    Function extracted from module MetagenomicDC/models/CNN.py of
    MetagenomicDC package [Fiannaca et al. 2018]
    https://github.com/IcarPA-TBlab/MetagenomicDC/blob/master/models/CNN.py
    """

    model = Sequential()
    model.add(Conv1D(5,5, input_shape = (nb_features, 1), padding = 'valid')) #input_dim
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size = 2, padding = 'valid'))
    model.add(Conv1D(10, 5, padding = 'valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size = 2, padding = 'valid'))
    # MLP
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax', dtype = 'float32'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'], jit_compile = True)

    return model

# Wide convolutional multiclass classifier
def build_wideCNN(nb_features, nb_classes):
    """
    Function adapted in keras from module CHEER/Classifier/model/Wcnn.py of
    CHEER package [Shang et al. 2021]
    https://github.com/KennthShang/CHEER/blob/master/Classifier/model/Wcnn.py
    """

    inputs = Input(shape = (nb_features, 1))
    # embed = Embedding(248, 100)(inputs)
    # inputs = Reshape((nb_features, -1, 1))(inputs)

    conv1 = Conv1D(256, 3, activation = 'relu')(inputs)
    conv1 = MaxPooling1D(pool_size = 1, strides = nb_features)(conv1)

    conv2 = Conv1D(256, 7, activation = 'relu')(inputs)
    conv2 = MaxPooling1D(pool_size = 1, strides = nb_features)(conv2)

    conv3 = Conv1D(256, 11, activation = 'relu')(inputs)
    conv3 = MaxPooling1D(pool_size = 1, strides = nb_features)(conv3)

    conv4 = Conv1D(256, 15, activation = 'relu')(inputs)
    conv4 = MaxPooling1D(pool_size = 1, strides = nb_features)(conv4)

    net = Concatenate(axis = 1)([conv1,conv2,conv3,conv4])
    net = Flatten()(net)

    net = Dense(512)(net)
    net = Activation("relu")(net)
    net = Dropout(0.5)(net)

    net = Dense(nb_classes)(net)
    outputs = Activation('softmax', dtype = 'float32')(net)
    
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'], jit_compile = True)

    return model
