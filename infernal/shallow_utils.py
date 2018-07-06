# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

"""
Utility functions for shallow models
"""

import os
import numpy as np
from six.moves import cPickle


CLASSIFIER_FILENAME = 'classifier.pickle'
NORMALIZER_FILENAME = 'normalizer.pickle'
TRANSFORMER_FILENAME = 'transformer.pickle'


def load_data(filename):
    """
    Load the data from the given path

    :return: a 2d array x and a 1d array y
    """
    data = np.load(filename)
    x, y = data['x'], data['y']
    return x, y


def load_classifier(path):
    """
    Load a classifer serialized as a pickle from the given directory.
    """
    path = os.path.join(path, CLASSIFIER_FILENAME)
    with open(path, 'rb') as f:
        classifier = cPickle.load(f)

    return classifier


def load_transformer(path):
    """
    Load a serialized transformer (such as a PCA object) from a pickle.
    """
    path = os.path.join(path, TRANSFORMER_FILENAME)
    if not os.path.isfile(path):
        return None

    with open(path, 'rb') as f:
        transformer = cPickle.load(f)

    return transformer


def load_normalizer(path):
    """
    Load a normalizer serialized as a pickle from the given directory.
    """
    path = os.path.join(path, NORMALIZER_FILENAME)
    if not os.path.isfile(path):
        # no normalizer was created for this model
        return None

    with open(path, 'rb') as f:
        normalizer = cPickle.load(f)

    return normalizer


def save(path, classifier, normalizer=None, transformer=None):
    """
    Save the classifier and the normalizer to the given directory.
    """
    classifier_filename = os.path.join(path, CLASSIFIER_FILENAME)
    with open(classifier_filename, 'wb') as f:
        cPickle.dump(classifier, f)

    if normalizer:
        normalizer_filename = os.path.join(path, NORMALIZER_FILENAME)
        with open(normalizer_filename, 'wb') as f:
            cPickle.dump(normalizer, f)

    if transformer:
        transformer_filename = os.path.join(path, TRANSFORMER_FILENAME)
        with open(transformer_filename, 'wb') as f:
            cPickle.dump(transformer, f)
