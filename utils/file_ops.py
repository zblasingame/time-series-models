"""Module for loading and parsing training data.

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import csv
import numpy as np


def load_data(filename):
    """Returns the features of a dataset.

    Args:
        filename (str): File location (csv formatted).

    Returns:
        tuple of np.ndarray: Tuple consisiting of the features,
            X, and the labels, Y.
    """

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = np.array([row for row in reader
                         if '#' not in row[0]]).astype(np.float32)

    X = data[:, 1:]
    Y = data[:, 0]

    Y = np.clip(Y, 0, 1)

    return X, Y


def batcher(data, batch_size=100):
    """Creates a generator to yield batches of batch_size.
    When batch is too large to fit remaining data the batch
    is clipped.

    Args:
        data (List of np.ndarray): List of data elements to be batched.
            The first dimension must be the batch size and the same
            for all data elements.
        batch_size (int = 100): Size of the mini batches.
    Yields:
        The next mini_batch in the dataset.
    """

    batch_start = 0
    batch_end   = batch_size

    while batch_end < data[0].shape[0]:
        yield [el[batch_start:batch_end] for el in data]

        batch_start = batch_end
        batch_end   += batch_size

    yield [el[batch_start:] for el in data]


def normalize(data, _min, _max):
    """Function to normalize a dataset of features

    Args:
        data (np.ndarray):
            Feature matrix.
        _min (list):
            List of minimum values per feature.
        _max (list):
            List of maximum values per feature.

    Returns:
        (np.ndarray): Normalized features of the same shape as data
    """

    new_data = (data - _min) / (_max - _min)

    # check if feature is constant, will be nan in new_data
    np.place(new_data, np.isnan(new_data), 1)

    return new_data
