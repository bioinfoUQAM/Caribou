
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
        self._iter_generator()

    # Shuffle list of positions of array
    def _shuffle(self):
        # Update indexes and shuffle position after each epoch
        self.positions_list = np.arange(len(self.positions_list))
        if self.shuffle:
            np.random.shuffle(self.positions_list)
            if self.cv:
                self.training_length = int(self.len_array*0.8)
                self.training_positions = self.positions_list[:self.training_length]
                self.testing_positions = self.positions_list[self.training_length:]

    def _iter_generator(self):
        if self.cv:
            self.iterator_train = self.iter_minibatch(self.training_positions)
            self.iterator_test = self.iter_minibatch(self.testing_positions)
        else:
            self.iterator = self.iter_minibatch(self.positions_list)

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
    def __init__(self, array, labels, positions_list, batch_size, kmers, ids, nb_classes, cv = 0, shuffle = True):
        # Initialization
        self.handle = tb.open_file(array, "r")
        try:
            self.array = self.handle.root.scaled
        except:
            self.array = self.handle.root.data
        self.labels = labels
        self.batch_size = batch_size
        self.kmers = kmers
        self.nb_classes = nb_classes
        self.cv = cv
        self.shuffle = shuffle
        #
        self.len_array = self.array.nrows
        self.positions_list = list(positions_list)
        self.list_IDs = ids

    def __data_generation(self, list_pos_temp):
        # Generate data
        # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, len(self.kmers)))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, pos in enumerate(list_pos_temp):
            # Store sample
            X[i,] = self.array.read()[pos,]

            # Store class
            y[i] = self.labels[pos]

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

def iter_generator_keras(array, labels, batch_size, kmers, ids, nb_classes, cv = 0, shuffle = True, training = True):
    params = {'batch_size': batch_size,
              'nb_classes': nb_classes,
              'kmers': kmers,
              'ids': ids,
              'cv': cv,
              'shuffle': shuffle}

    if cv and training:
        positions_list = np.array(range(len(labels)))
        np.random.shuffle(positions_list)
        training_length = int(len(labels)*0.8)
        validating_length = int(training_length*0.2)
        training_length = int(training_length*0.8)
        training_positions = positions_list[:training_length]
        validating_positions = positions_list[training_length:(training_length + validating_length)]
        testing_positions = positions_list[(training_length + validating_length):]

        iterator_train = DataGeneratorKeras(array, labels, training_positions, **params)
        iterator_val = DataGeneratorKeras(array, labels, validating_positions, **params)
        iterator_test = DataGeneratorKeras(array, labels, testing_positions, **params)

        return iterator_train, iterator_val, iterator_test

    elif not cv and training:
        positions_list = np.array(range(len(labels)))
        np.random.shuffle(positions_list)
        training_length = int(len(labels)*0.8)
        training_positions = positions_list[:training_length]
        validating_positions = positions_list[training_length:]

        iterator_train = DataGeneratorKeras(array, labels, training_positions, **params)
        iterator_val = DataGeneratorKeras(array, labels, validating_positions, **params)
        return iterator_train, iterator_val

    else:
        positions_list = np.array(range(len(labels)))
        iterator = DataGeneratorKeras(array, labels, positions_list, **params)
        return iterator
