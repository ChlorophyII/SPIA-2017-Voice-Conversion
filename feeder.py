import os, os.path
import numpy as np
import random
import string
'''
possible phases: 'training', 'validation', 'test', 'conversion'
'''

class feeder(object):
    def __init__(self, batch_size, path, n_steps, n_inputs, phase='training'):
        self._phase = phase
        self._path = path
        self._n_steps = n_steps
        self._n_inputs = n_inputs
        self._filenames = [name for name in os.listdir(self._path) if os.path.isfile(self._path+name) and '.dat' in name and '.csv' not in name]
        self._data = {}
        if self._phase == 'conversion':
            self._n_data = len(self._filenames)
            self._batch_size = self._n_data
            for filename in self._filenames:
                padding = np.zeros([self._n_steps, self._n_inputs])
                data = np.transpose(np.genfromtxt(self._path+filename, delimiter=','))[:,1:]
                padding[:np.shape(data)[0],:] = data
                self._data[filename] = padding
                weight = np.zeros([self._n_steps, self._n_inputs])
                weight[:np.shape(data)[0],:] = np.ones_like(data)
                self._data[filename+'_weight'] = weight
                self._data[filename+'_shape'] = np.shape(data)
        else: # phase == 'training' or 'validation' or 'test'
            assert len(self._filenames) % 2 == 0, \
                "Number of sources and number of targets do not match"
            self._n_data = len(self._filenames) / 2
            
            if self._phase == 'training': # for training, a batch is all the data in a folder
                self._batch_size = batch_size
                assert self._batch_size <= self._n_data, \
                    "Batch_size %r should be smaller than number of data %r" % \
                    (self._batch_size, self._n_data)
                self._all_data_source = []
                self._all_data_target = []
            else:
                self._batch_size = self._n_data

            self._pair_filenames = [[s, t] for s, t in zip(self._filenames[0::2], self._filenames[1::2])]
            self._epoch_filenames = []
            i = 0
            for filename in self._filenames:
                padding = np.zeros([self._n_steps, self._n_inputs])
                data = np.transpose(np.genfromtxt(self._path+filename, delimiter=','))[:,1:]
                if phase == 'training':
                    if i % 2 == 0:
                        if self._all_data_source == []:
                            self._all_data_source = data
                        else:
                            self._all_data_source = np.append(self._all_data_source, data, axis=0)
                    else:
                        if self._all_data_target == []:
                            self._all_data_target = data
                        else:
                            self._all_data_target = np.append(self._all_data_target, data, axis=0)

                
                padding[:np.shape(data)[0],:] = data
                self._data[filename] = padding
                if i % 2 == 0:
                    weight = np.zeros([self._n_steps, self._n_inputs])
                    weight[:np.shape(data)[0],:] = np.ones_like(data)
                    self._data[filename+'_weight'] = weight
                    self._data[filename+'_shape'] = np.shape(data)
                i += 1
            if phase == 'training':
                self._mean_source = np.mean(self._all_data_source, axis=0)
                self._std_source = np.std(self._all_data_source, axis=0)
                self._all_data_source = None
                self._mean_target = np.mean(self._all_data_target, axis=0)
                self._std_target = np.std(self._all_data_target, axis=0)
                self._all_data_target = None

    @property
    def n_data(self):
        return self._n_data

    @property
    def mean(self):
        assert self._phase == 'training', \
            "\"mean\" should only be retrieved in training data."
        mean = np.append([[self._mean_source]], [[self._mean_target]], axis=0)
        return mean

    @property
    def std(self):
        assert self._phase == 'training', \
            "\"std\" should only be retrieved in training data."
        return np.append([[self._std_source]], [[self._std_target]], axis=0)

    def __get_batch_filenames(self):
        if self._phase != 'conversion':
            if len(self._epoch_filenames) < self._batch_size:
                batch_filenames = self._epoch_filenames
                self._epoch_filenames = self._pair_filenames
                rest = random.sample(self._epoch_filenames, int(self._batch_size-len(batch_filenames)))
                batch_filenames = batch_filenames + rest
                self._epoch_filenames = [name for name in self._epoch_filenames \
                                         if name not in rest]
            else:
                batch_filenames = random.sample(self._epoch_filenames, int(self._batch_size))
                self._epoch_filenames = [name for name in self._epoch_filenames \
                                         if name not in batch_filenames]
            return batch_filenames
        else:
            return self._filenames

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
        if self._phase != 'conversion':
            for [source_name, target_name] in batch_filenames:
                sources.append(self._data[source_name])
                targets.append(self._data[target_name])
                weights.append(self._data[source_name+'_weight'])
            return sources, targets, weights, batch_filenames
        else:
            for filename in batch_filenames:
                sources.append(self._data[filename])
                weights.append(self._data[filename+'_weight'])
            return sources, sources, weights, batch_filenames
        
    def save_outputs(self, outputs, batch_filenames):
        assert self._phase == 'conversion', \
            "Only in \"conversion\" phase data need to be saved, isn't it?"
        i = 0
        for filename in batch_filenames:
            output_filename = self._path+filename.replace(".dat", "_converted.csv")
            with open(output_filename, 'wb') as output_file:
                num_rows, num_cols = self._data[filename+'_shape']
                np.savetxt(output_file, np.transpose(outputs[i,:num_rows,:num_cols]), delimiter=',')
            i += 1
