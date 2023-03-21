
from keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Concatenate, Flatten, Attention, Activation, Bidirectional, Reshape

from models.kerasTF.attentionLayer import AttentionWeightedAverage

__author__ = "Nicolas de Montigny"

__all__ = ['build_attention','build_LSTM','build_deepLSTM','build_LSTM_attention','build_CNN','build_wideCNN']

# Self-aware binary classifier
def build_attention(nb_kmers):
    """
    Function extracted from module virnet/NNClassifier.py of
    VirNet package [Abdelkareem et al. 2018]
    https://github.com/alyosama/virnet/blob/master/NNClassifier.py
    """
    inputs = Input(shape = (nb_kmers,))
    x = Embedding(nb_kmers, 128)(inputs)

    x = LSTM(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1 )(x)
    x = LSTM(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1 )(x)
    x = AttentionWeightedAverage()(x)

    x = Dense(128, activation = "relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation = "tanh")(x)

    model = Model(inputs = inputs, outputs = x)
    model.compile(loss = BinaryCrossentropy(from_logits = True), optimizer = 'adam', metrics = ['accuracy'])

    return model

# Recurrent binary classifier
def build_LSTM(nb_kmers):
    """
    Function extracted from module seeker/train_model/train_model.py of
    Seeker package [Auslander et al. 2020]
    https://github.com/gussow/seeker/blob/master/train_model/train_model.py
    """
    
    inputs = Input(shape = (nb_kmers,))
    x = Embedding(nb_kmers, 128)(inputs)

    x = LSTM(128, recurrent_dropout = 0.1, dropout = 0.1)(x)

    x = Dense(1, activation = 'tanh')(x)
    
    model = Model(inputs = inputs, outputs = x)
    model.compile(loss=BinaryCrossentropy(from_logits = True), optimizer='adam', metrics=['accuracy'])

    return model

# Deep recurrent binary classifier
def build_deepLSTM(nb_kmers):
    """
    Function adapted from module deeplasmid/classifier/dl/DL_Model.py of
    Deeplasmid package [Andreopoulos et al. 2021]
    https://github.com/wandreopoulos/deeplasmid/blob/docker/classifier/dl/DL_Model.py
    """

    inputs = Input(shape=(nb_kmers,))

    netA = Embedding(nb_kmers, 128)(inputs)
    netA = LSTM(40, activation='tanh',recurrent_dropout=0.05,dropout=0.1,name='A_%d'%40,return_sequences=True) (netA)
    netA = LSTM(40, activation='tanh',recurrent_dropout=0.05,dropout=0.1,name='B_%d'%40) (netA)

    netB = Dense(100, activation='tanh',name='G_%d'%40) (inputs)
    netB = Dense(40, activation='tanh',name='H_%d'%40) (netB)

    net = Concatenate()([netA,netB])

    net = Dense(200, activation='relu', name='C_%d'%(10*2))(net)
    net = Dropout(0.1,name='fr_%.1f'%0.1)(net)

    net = Dense(10, activation='relu', name='D_%d'%10)(net)
    net = Dropout(0.1,name='fr_same')(net)

    outputs = Dense(1, activation='tanh', name='score')(net)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=BinaryCrossentropy(from_logits = True), optimizer='adam', metrics=['accuracy'])

    return model

# Recurrent self-aware multiclass classifier
def build_LSTM_attention(nb_kmers, nb_classes):
    """
    Function adapted in keras from module DeepMicrobes/models/embed_lstm_attention.py and
    default values for layers in script DeepMicrobes/models/define_flags.py of
    DeepMicrobes package [Liang et al. 2020]
    https://github.com/MicrobeLab/DeepMicrobes/blob/master/models/embed_lstm_attention.py
    """

    inputs = Input(shape = (nb_kmers,))
    net = Embedding(nb_kmers, 100)(inputs)
    net = Bidirectional(LSTM(300, return_sequences=True))(net)
    net = Attention(dropout = 0.2)([net,net])
    # MLP
    net = Dense((nb_kmers * 300 * 2), activation = 'relu')(net)
    net = Dropout(0.2)(net)
    net = Dense(nb_classes, activation = 'relu')(net)
    net = Dropout(0.2)(net)
    net = Flatten()(net)
    net = Dense(nb_classes)(net)
    outputs = Activation('softmax')(net)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss=CategoricalCrossentropy(from_logits = True), optimizer='adam', metrics=['accuracy'])

    return model

# Convolutional multiclass classifier
def build_CNN(nb_kmers, nb_classes):
    """
    Function extracted from module MetagenomicDC/models/CNN.py of
    MetagenomicDC package [Fiannaca et al. 2018]
    https://github.com/IcarPA-TBlab/MetagenomicDC/blob/master/models/CNN.py
    """

    model = Sequential()
    model.add(Conv1D(5,5, input_shape = (nb_kmers, 1), padding = 'valid')) #input_dim
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
    model.add(Activation('softmax'))
    model.compile(loss=CategoricalCrossentropy(from_logits = True), optimizer='adam', metrics=['accuracy'])

    return model

# Wide convolutional multiclass classifier
def build_wideCNN(nb_kmers, nb_classes):
    """
    Function adapted in keras from module CHEER/Classifier/model/Wcnn.py of
    CHEER package [Shang et al. 2021]
    https://github.com/KennthShang/CHEER/blob/master/Classifier/model/Wcnn.py
    """

    inputs = Input(shape = (nb_kmers,))
    embed = Embedding(248, 100)(inputs)
    embed = Reshape((nb_kmers, -1, 1))(embed)

    conv1 = Conv2D(256, 3, activation = 'relu')(embed)
    conv1 = MaxPooling2D(pool_size = (1,1), strides = nb_kmers)(conv1)

    conv2 = Conv2D(256, 7, activation = 'relu')(embed)
    conv2 = MaxPooling2D(pool_size = (1,1), strides = nb_kmers)(conv2)

    conv3 = Conv2D(256, 11, activation = 'relu')(embed)
    conv3 = MaxPooling2D(pool_size = (1,1), strides = nb_kmers)(conv3)

    conv4 = Conv2D(256, 15, activation = 'relu')(embed)
    conv4 = MaxPooling2D(pool_size = (1,1), strides = nb_kmers)(conv4)

    net = Concatenate(axis = 1)([conv1,conv2,conv3,conv4])
    net = Flatten()(net)

    net = Dense(512)(net)
    net = Activation("relu")(net)
    net = Dropout(0.5)(net)

    net = Dense(nb_classes)(net)
    outputs = Activation('softmax')(net)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss=CategoricalCrossentropy(from_logits = True), optimizer='adam', metrics=['accuracy'])

    return model
