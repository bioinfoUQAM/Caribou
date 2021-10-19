
import numpy as np
import pandas as pd
import tables as tb

import keras

__author__ = "nicolas"

# Data
class DataGenerator():
    #Initialization
    def __init__(self, array, labels, batch_size, kmers, nb_classes, cv = 0, shuffle = True):
        handle = tb.open_file(array, "r")
        self.array = handle.root.data
        self.labels = labels
        self.batch_size = batch_size
        self.kmers = kmers
        self.nb_classes = nb_classes
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
            self.iterator_train = self.iter_minibatch()
            self.iterator_test = self.iter_minibatch()
        else:
            self.iterator = self.iter_minibatch()

    def iter_minibatch(self):
        start = 0
        end = self.batch_size
        while start < self.len_array:
            if start < (self.len_array - self.batch_size):
                X, y = self.get_minibatch(self.positions_list[start:end])
                yield X, y
                start = start + self.batch_size
                end = end + self.batch_size
            else:
                X, y = self.get_minibatch(self.positions_list[start:end])
                yield X, y
                start = self.len_array

    def get_minibatch(self, positions):
        X = pd.DataFrame(self.array.read()[positions[0]:positions[-1]], index = self.kmers["ids"][positions[0]:positions[-1]], columns = self.kmers["kmers_list"])
        y = self.labels.iloc[positions[0]:positions[-1]][0]
        return X, y

class DataGeneratorKeras(keras.utils.Sequence):
    def __init__(self, array, positions_list, labels, batch_size, kmers, nb_classes, cv = 0, shuffle = True):
        # Initialization
        handle = tb.open_file(array, "r")
        self.array = handle.root.data
        self.labels = labels
        self.batch_size = batch_size
        self.kmers = kmers
        self.nb_classes = nb_classes
        self.cv = cv
        self.shuffle = shuffle
        #
        self.len_array = array.nrows
        self.positions_list = positions_list
        self.list_IDs = kmers["ids"]
        #
        self.shuffle()
        iter_generator()
        handle.close()

    def shuffle(self):
        # Update indexes and shuffle position after each epoch
        self.positions_list = np.arange(len(self.positions_list))
        if self.shuffle:
            np.random.shuffle(self.positions_list)

    def __data_generation(self, list_IDs_temp):
        # Generate data
        # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, len(self.kmer["kmers_list"])))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_pos_temp):
            # Store sample
            X[i,:] = self.array.read()[ID,:]

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_labels=self.n_labels)

    def __len__(self):
        # Batches/epoch
        return int(np.floor(self.len_array / self.batch_size))

    def __getitem__(self, index):
        # Generate a batch of data + indexes
        indexes = self.positions_list[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_pos_temp = [self.position_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_pos_temp)

        return X, y

# #####
# Data build functions
# ####################

def iter_generator_keras(array, labels, batch_size, kmers, nb_classes, cv = 0, shuffle = True):
    params = {'batch_size': batch_size,
                'n_classes': nb_classes,
                'kmers': kmers,
                'cv': cv,
                'shuffle': shuffle}

    if cv:

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

    else:

        positions_list = np.array(range(len(labels)))
        np.random.shuffle(positions_list)
        training_length = int(len(labels)*0.8)
        training_positions = positions_list[:training_length]
        validating_positions = positions_list[training_length:]

        iterator_train = DataGeneratorKeras(array, labels, training_positions, **params)
        iterator_val = DataGeneratorKeras(array, labels, validating_positions, **params)
        return iterator_train, iterator_val
