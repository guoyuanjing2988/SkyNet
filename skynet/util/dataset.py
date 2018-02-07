import numpy as np

class Dataset(object):

    def __init__(self, vectors):
        self._vectors = vectors
        self._num_examples = len(vectors)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def vectors(self):
        return self._vectors

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def fetch_next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch

        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._vectors = self._vectors[perm0]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1

            remaining_num_examples = self._num_examples - start
            remaining_vec = self._vectors[start : self._num_examples]

            if shuffle:
                perm0 = np.arange(self._num_examples)
                np.random.shuffle(perm0)
                self._vectors = self._vectors[perm0]

            start = 0
            self._index_in_epoch = batch_size - remaining_num_examples
            end = self._index_in_epoch
            new_vec = self._vectors[start : end]
            return np.concatenate((remaining_vec, new_vec), axis=0)

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._vectors[start : end]
