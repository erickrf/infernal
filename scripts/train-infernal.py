# -*- coding: utf-8 -*-

"""
Analyze the relevancy of each feature with respect to entailment classes.
"""

from __future__ import division, print_function, unicode_literals

import argparse

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from infernal import shallow_utils


def get_classifier(name, balanced):
    class_weight = 'balanced' if balanced else None
    if name == 'log-regression':
        return LogisticRegression(class_weight=class_weight)
    elif name == 'svm':
        return SVC(class_weight=class_weight)
    elif name == 'random-forest':
        return RandomForestClassifier(500, max_features='sqrt',
                                      class_weight=class_weight)
    elif name == 'grad-boost':
        return GradientBoostingClassifier(n_estimators=500,
                                          max_features='sqrt',
                                          learning_rate=0.01)
    elif name == 'naive-bayes':
        return GaussianNB()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train_data', help='Training data (npz archive)')
    parser.add_argument('classifier', help='Type of classifier',
                        choices=['log-regression', 'svm', 'random-forest',
                                 'grad-boost'])
    parser.add_argument('output', help='Directory to save the model')
    parser.add_argument('-b', dest='balanced', action='store_true',
                        help='Use balanced class weights')
    parser.add_argument('-s', action='store_true', dest='scaler',
                        help='Use scaler (centers and normalizes features)')
    args = parser.parse_args()

    x, y = shallow_utils.load_data(args.train_data)
    if args.scaler:
        scaler = RobustScaler()
        x = scaler.fit_transform(x)
    else:
        scaler = None

    c = get_classifier(args.classifier, args.balanced)
    c.fit(x, y)

    shallow_utils.save(args.output, c, scaler)
