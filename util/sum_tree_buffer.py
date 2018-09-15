import numpy as np


class SumTreeBuffer(object):

    def __init__(self, size=100000, dtype=object):
        self._data = np.zeros(size, dtype=dtype)
        self._size = self._data.size
        self._tree_size = 2 * self._size - 1
        self._tree = np.zeros(self._tree_size, dtype=np.float32)
        self._data_idx = 0
        self.full = False

    def _propogate(self, idx, difference):
        parent_idx = (idx - 1) // 2
        self._tree[parent_idx] += difference

        if parent_idx > 0:
            self._propogate(parent_idx, difference)

    def _get_idx(self, idx, value):
        left = 2 * idx + 1
        right = left + 1

        if left >= self._tree_size:
            return idx

        left_val = self._tree[left]
        if value <= left_val:
            return self._get_idx(left, value)
        else:
            return self._get_idx(right, value - left_val)

    def update(self, idx, value):
        difference = value - self._tree[idx]
        self._tree[idx] = value
        self._propogate(idx, difference)

    def total(self):
        return self._tree[0]

    def add(self, value, data):
        idx = self._data_idx + self._size - 1
        self._data[self._data_idx] = data
        self.update(idx, value)

        self._data_idx += 1
        if self._data_idx >= self._size:
            self._data_idx = 0
            self.full = True

    def extend(self, other):
        if other.full:
            max_idx = other._size
        else:
            max_idx = other._data_idx + 1
        idx_diff = other._size - 1
        for i in range(max_idx):
            self.add(other._tree[i + idx_diff], other._data[i])

    def get_index(self, value):
        return self._get_idx(0, value)

    def get_at_index(self, idx):
        return self._data[idx]

    def get_entry(self, value):
        idx = self.get_index(value)
        entry = self.get_at_index(idx - self._size + 1)
        return idx, entry
