# -*- coding: utf-8 -*-

"""
Evaluate a shallow classifier
"""

from __future__ import division, print_function, unicode_literals

import argparse
import numpy as np
from sklearn.metrics import f1_score

from infernal import shallow_utils as shallow


def evaluate(classifier, normalizer, transformer, x, y):
    """
    Evaluate the performance of the classifier with the given data
    """
    if normalizer is not None:
        x = normalizer.transform(x)

    if transformer is not None:
        x = transformer.transform(x)

    preds = classifier.predict(x)
    acc = np.sum(y == preds) / len(y)
    f1 = f1_score(y, preds, average='macro')

    print('Accuracy: {:.2%}'.format(acc))
    print('F1 macro: {:.3}'.format(f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', help='Preprocessed data (npz) to evaluate the'
                                     'classifier on')
    parser.add_argument('model', help='Directory with saved model')
    args = parser.parse_args()

    x, y = shallow.load_data(args.data)
    classifier = shallow.load_classifier(args.model)
    normalizer = shallow.load_normalizer(args.model)
    transformer = shallow.load_transformer(args.model)
    evaluate(classifier, normalizer, transformer, x, y)
