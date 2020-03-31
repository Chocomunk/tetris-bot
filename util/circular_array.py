import numpy as np


class CircularArray(object):

    def __init__(self, size, init_array=None, dtype=np.float32):
        if init_array:
            if len(init_array) > size:
                self._data = np.array(init_array[:size], dtype=dtype)
                self._size = size
            else:
                data = init_array
                self._size = len(init_array)
                for i in range(size - len(init_array)):
                    data.append(0)
                self._data = np.array(data, dtype=dtype)
        else:
            self._data = np.zeros(size, dtype=dtype)

        self._capacity = size
        self._index = 0

    def add(self, entry):
        self._data[self._index] = entry
        self._index += 1
        if self._index >= self._capacity:
            self. _index = 0

    def mean(self):
        return np.mean(self._data)

    def __len__(self):
        return self._size
