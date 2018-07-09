# -*- coding: utf-8 -*-

"""
"""

from __future__ import division, print_function, unicode_literals

import argparse
from six.moves import cPickle

from infernal import openwordnetpt as own

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='OpenWordNet-PT file with .nt extension')
    parser.add_argument('output', help='Name to save the pickled file')
    args = parser.parse_args()

    own.load_wordnet(args.input)
    with open(args.output, 'wb') as f:
        cPickle.dump(own._wn_graph, f)
