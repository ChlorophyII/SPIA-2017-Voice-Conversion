import os, os.path
import numpy as np
import random

class feeder(object):
    def __init__(self, batch_size, path, n_steps, n_inputs, phase='training'):
        self._path = path
        self._n_steps = n_steps
        self._n_inputs = n_inputs
        self._filenames = [name for name in os.listdir(self._path) if os.path.isfile(self._path+name) and '.dat' in name]
        assert len(self._filenames) % 2 == 0, \
            "Number of sources and number of targets do not match"
        self._n_data = len(self._filenames) / 2
        if batch_size == -1: # for training, a batch is all the data in a folder
            self._batch_size = self._n_data
        else:
            self._batch_size = batch_size

        assert self._batch_size <= self._n_data, \
            "Batch_size %r should be smaller than number of data %r" % \
            (self._batch_size, self._n_data)
        self._pair_filenames = [[s, t] for s, t in zip(self._filenames[0::2], self._filenames[1::2])]
        self._epoch_filenames = []
    
    @property
    def n_data(self):
        return self._n_data

    def __get_batch_filenames(self):
        if len(self._epoch_filenames) < self._batch_size:
            batch_filenames = self._epoch_filenames
            self._epoch_filenames = self._pair_filenames
            rest = random.sample(self._epoch_filenames, self._batch_size-len(batch_filenames))
            batch_filenames = batch_filenames + rest
            self._epoch_filenames = [name for name in self._epoch_filenames \
                                     if name not in rest]
        else:
            batch_filenames = random.sample(self._epoch_filenames, self._batch_size)
            self._epoch_filenames = [name for name in self._epoch_filenames \
                                     if name not in batch_filenames]
        return batch_filenames

    def get_batch(self):
        """
            Outputs:
            sources: A 3-d numpy array of shape [batch_size, n_steps, n_inputs]
            targets: A 3-d numpy array of shape [batch_size, n_steps, n_inputs]
        """
        batch_filenames = self.__get_batch_filenames()
        sources = []
        targets = []
        weights = []
        for [source_name, target_name] in batch_filenames:
            source = np.zeros([self._n_steps, self._n_inputs])
            s = np.transpose(np.genfromtxt(self._path+source_name, delimiter=','))
            s = s[:,1:]
            source[:np.shape(s)[0],:] = s
            target = np.zeros([self._n_steps, self._n_inputs])
            t = np.transpose(np.genfromtxt(self._path+target_name, delimiter=','))
            t = t[:,1:]
            target[:np.shape(t)[0],:] = t
            weight = np.zeros([self._n_steps, self._n_inputs])
            weight[:np.shape(t)[0],:] = np.ones_like(t)
            sources.append(source)
            targets.append(target)
            weights.append(weight)
        return sources, targets, weights, batch_filenames
