
import numpy as np
import pandas as pd
import tables as tb

from tensorflow.keras.utils import Sequence, to_categorical

__author__ = "nicolas"

# Data
class DataGenerator():
    #Initialization
    def __init__(self, array, labels, batch_size, kmers, ids, cv = 0, shuffle = True):
        self.handle = tb.open_file(array, "r")
        try:
            self.array = self.handle.root.scaled
        except:
            self.array = self.handle.root.data
        self.labels = labels
        self.batch_size = batch_size
        self.kmers = kmers
        self.ids = ids
        self.cv = cv
        self.shuffle = shuffle
        #
        self.len_array = self.array.nrows
        self.training_length = 0
        self.training_positions = []
        self.testing_positions = []
        self.positions_list = np.arange(self.len_array)
        #
        self._shuffle()
        self.iterator = self.iter_minibatch(self.positions_list)

    # Shuffle list of positions of array
    def _shuffle(self):
        # Shuffle position after each epoch
        if self.shuffle:
            np.random.shuffle(self.positions_list)

    def iter_minibatch(self, positions):
        start = 0
        end = self.batch_size
        while start < len(positions):
            if start < (len(positions) - self.batch_size):
                X, y = self.get_minibatch(positions[start:end])
                yield X, y
                start = start + self.batch_size
                end = end + self.batch_size
            else:
                X, y = self.get_minibatch(positions[start:self.len_array])
                yield X, y
                start = len(positions)

    def get_minibatch(self, positions):
        X = pd.DataFrame(columns = self.kmers)
        y = pd.DataFrame()
        index_list = []
        for pos in positions:
            X = X.append(pd.Series(self.array.read()[pos], index = self.kmers), ignore_index = True)
            y = y.append(pd.Series(self.labels[pos]), ignore_index = True)
            index_list.append(self.ids[pos])
        X.index = index_list
        y.index = index_list

        return X, y

class DataGeneratorKeras(Sequence):
    def __init__(self, array, labels, positions_list, batch_size, kmers, ids, classifier, shuffle = True):
        # Initialization
        self.handle = tb.open_file(array, "r")
        try:
            self.array = self.handle.root.scaled
        except:
            self.array = self.handle.root.data
        self.labels = labels
        self.batch_size = batch_size
        self.kmers = kmers
        self.classifier = classifier
        self.shuffle = shuffle
        #
        self.len_array = self.array.nrows
        self.positions_list = list(positions_list)
        self.list_IDs = ids
        self.on_epoch_end()

    # Shuffle list of positions of array
    def on_epoch_end(self):
        # Shuffle position after each epoch
        if self.shuffle:
            np.random.shuffle(self.positions_list)

    def __data_generation(self, list_pos_temp):
        X = np.empty((self.batch_size, len(self.kmers)))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, pos in enumerate(list_pos_temp):
            # Store sample
            X[i,] = self.array.read()[pos,]

            # Store class
            y[i] = self.labels[pos]


        if self.classifier in ["lstm","deeplstm"]:
            X = X.reshape(1, self.batch_size, len(self.kmers))
            y = y.reshape(1, self.batch_size, 1)

        return X, y

    def __len__(self):
        # Batches/epoch
        return int(np.floor(len(self.positions_list) / self.batch_size))

    def __getitem__(self, index):
        # Generate a batch of data + indexes
        list_pos_temp = self.positions_list[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(list_pos_temp)

        return X, y

# ####################
# Data build functions
# ####################
def iter_generator(array, labels, batch_size, kmers, ids, cv, shuffle = True, training = True):
    if cv and training:
        positions_list = np.array(range(len(labels)))
        np.random.shuffle(positions_list)

        training_length = int(len(labels)*0.8)

        training_positions = positions_list[:training_length]
        testing_positions = positions_list[training_length:]

        iterator_train = DataGenerator(array, labels, batch_size, kmers, ids, cv, shuffle)
        iterator_test = DataGenerator(array, labels, batch_size, kmers, ids, cv, shuffle)

        return iterator_train, iterator_test

    else:
        positions_list = np.array(range(len(labels)))
        iterator = DataGenerator(array, labels, batch_size, kmers, ids, cv, shuffle)
        return iterator

def iter_generator_keras(array, labels, batch_size, kmers, ids, cv, classifier, shuffle, training):

    if cv and training:
        positions_list = np.array(range(len(labels)))
        np.random.shuffle(positions_list)

        training_length = int(len(labels)*0.8)
        validating_length = int(training_length*0.2)
        training_length = int(training_length*0.8)

        training_positions = positions_list[:training_length]
        validating_positions = positions_list[training_length:(training_length + validating_length)]
        testing_positions = positions_list[(training_length + validating_length):]

        iterator_train = DataGeneratorKeras(array, labels, training_positions, batch_size, kmers, ids, classifier, shuffle = True)
        iterator_val = DataGeneratorKeras(array, labels, validating_positions, batch_size, kmers, ids, classifier, shuffle = True)
        iterator_test = DataGeneratorKeras(array, labels, testing_positions, 1, kmers, ids, classifier, shuffle = False)

        return iterator_train, iterator_val, iterator_test

    elif not cv and training:
        positions_list = np.array(range(len(labels)))
        np.random.shuffle(positions_list)
        training_length = int(len(labels)*0.8)
        training_positions = positions_list[:training_length]
        validating_positions = positions_list[training_length:]

        iterator_train = DataGeneratorKeras(array, labels, training_positions, batch_size, kmers, ids, classifier, shuffle = True)
        iterator_val = DataGeneratorKeras(array, labels, validating_positions, batch_size, kmers, ids, classifier, shuffle = True)
        return iterator_train, iterator_val

    else:
        positions_list = np.array(range(len(labels)))
        iterator = DataGeneratorKeras(array, labels, positions_list, batch_size, kmers, ids, classifier, shuffle = False)
        return iterator
