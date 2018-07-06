# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from nltk.tokenize import RegexpTokenizer
import argparse
import os

"""
Script for tokenizing Portuguese text according to the Universal Dependencies
(UD) tokenization standards. This script was not created by the UD team; it was
based on observation of the corpus.
"""


def tokenize(text):
    """
    Tokenize the given sentence in Portuguese.
    :param text: text to be tokenized, as a string
    """
    tokenizer_regexp = r'''(?ux)
    # the order of the patterns is important!!
    # more structured patterns come first
    [a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+|    # emails
    (?:https?://)?\w{2,}(?:\.\w{2,})+(?:/\w+)*|                  # URLs
    (?:[\#@]\w+)|                     # Hashtags and twitter user names
    (?:[^\W\d_]\.)+|                  # one letter abbreviations, e.g. E.U.A.
    (?:[DSds][Rr][Aa]?)\.|            # common abbreviations such as dr., sr., sra., dra.
    (?:\B-)?\d+(?:[:.,]\d+)*(?:-?\w)*|
        # numbers in format 999.999.999,999, possibly followed by hyphen and alphanumerics
        # \B- avoids picks as F-14 as a negative number
    \.{3,}|                           # ellipsis or sequences of dots
    \w+|                              # alphanumerics
    -+|                               # any sequence of dashes
    \S                                # any non-space character
    '''
    tokenizer = RegexpTokenizer(tokenizer_regexp)

    return tokenizer.tokenize(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+',
                        help='Files to tokenize (new files with .token '
                             'extension will be generated)')
    args = parser.parse_args()

    for filename in args.inputs:
        print('Tokenizing %s' % filename)

        tokenized_lines = []
        basename, _ = os.path.splitext(filename)
        new_name = basename + '.token'
        with open(filename, 'rb') as f:
            for line in f:
                line = line.decode('utf-8')
                tokens = tokenize(line)
                tokenized_line = ' '.join(tokens)
                tokenized_lines.append(tokenized_line)

        text = '\n'.join(tokenized_lines)
        with open(new_name, 'wb') as f:
            f.write(text.encode('utf-8'))