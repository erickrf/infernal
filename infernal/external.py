# -*- coding: utf-8 -*-

from __future__ import unicode_literals

"""
Functions for calling external NLP tools.
"""

import json
import requests
from six.moves import urllib

from . import config

nlpnet_tagger = None


def call_corenlp(text, corenlp_depparse_path=None, corenlp_pos_path=None):
    """
    Call Stanford corenlp, which should be running at the address specified in
    the config module.

    Only a dependency parser and POS tagger are run.

    :param text: text with tokens separated by whitespace
    :param corenlp_depparse_path: if not using a .jar model saved on the same
        directory, specify the path to the dependency parser
    :param corenlp_pos_path: same as above for the POS tagger
    """
    properties = {'tokenize.whitespace': 'true',
                  'annotators': 'tokenize,pos,depparse',
                  'outputFormat': 'conllu',
                  'ssplit.eolonly': True}
    if corenlp_depparse_path:
        properties['depparse.model'] = corenlp_depparse_path

    if corenlp_pos_path:
        properties['pos.model'] = corenlp_pos_path

    # use json dumps function to convert the nested dictionary to a string
    properties_val = json.dumps(properties)
    params = {'properties': properties_val}

    # we encode the URL params using urllib because we need a URL with GET
    # parameters even though we are making a POST request. The POST data is the
    # text itself.
    encoded_params = urllib.parse.urlencode(params)
    url = '{url}:{port}/?{params}'.format(url=config.corenlp_url,
                                          port=config.corenlp_port,
                                          params=encoded_params)

    headers = {'Content-Type': 'text/plain;charset=utf-8'}
    response = requests.post(url, text.encode('utf-8'), headers=headers)
    response.encoding = 'utf-8'

    # bug: \0 character appears in the response
    output = response.text.replace('\0', '')

    return output
