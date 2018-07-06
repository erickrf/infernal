# -*- coding: utf-8 -*-

from __future__ import unicode_literals

"""
Conversion from number strings to ints.
"""

import logging
import re

values = {'zero': 0,
          'um': 1,
          'uma': 2,
          'dois': 2,
          'duas': 2,
          'três': 3,
          'quatro': 4,
          'cinco': 5,
          'seis': 6,
          'sete': 7,
          'oito': 8,
          'nove': 9,
          'dez': 10,
          'onze': 11,
          'doze': 12,
          'treze': 13,
          'quatorze': 14,
          'catorze': 14,
          'quinze': 15,
          'dezesseis': 16,
          'dezasseis': 16,
          'dezessete': 17,
          'dezassete': 17,
          'dezoito': 18,
          'dezenove': 19,
          'dezanove': 19,
          'vinte': 20,
          'trinta': 30,
          'quarenta': 40,
          'cinquenta': 50,
          'sessenta': 60,
          'setenta': 70,
          'oitenta': 80,
          'noventa': 90,
          'cem': 100,
          'duzentos': 200,
          'trezentos': 300,
          'quatrocentos': 400,
          'quinhentos': 500,
          'seiscentos': 600,
          'setecentos': 700,
          'oitocentos': 800,
          'novecentos': 900,
          'mil': 1000,
          'milhão': 1e6,
          'bilhão': 1e9,
          'trilhão': 1e12
}

multipliers = {'mil': 1000, 'milhar': 1000, 'milhares': 1000,
               'milhão': 1e6, 'milhões': 1e6,
               'bilhão': 1e9, 'bilhões': 1e9,
               'trilhão': 1e12, 'trilhões': 1e12}


def _get_number(text):

    if re.match(r'-?[.,\d]+$', text):
        en_format = text.replace('.', '').replace(',', '.')
        return float(en_format)

    try:
        return float(values[text.lower()])
    except KeyError:
        msg = "Can't convert this number to digits: {}".format(text)
        logging.warning(msg)
        return None


def get_number(tokens):
    """
    Return a number representation in this token. The value might be the
    content of this token itself or composed with its dependents.
    
    If it's not possible, raise a ValueError.
    """
    token = tokens[0]

    # if this is a composed number, other parts come as dependents
    other_parts = [dep for dep in token.dependents
                   if dep.dependency_relation == 'num' or dep.pos == 'NUM']

    if len(other_parts):
        other_value = get_number(other_parts)
    else:
        other_value = None

    # if this is a "multiplier" value such as mil / milhão / bilhão etc,
    # just multiply the other parts
    # if it's not, sum them.
    text = token.text
    if text in multipliers and other_value is not None:
        multiplier_value = multipliers[text]
        return multiplier_value * other_value

    value = _get_number(text)
    if value is None:
        return None

    # here we treat cases like vinte e cinco
    if other_value is not None:
        return value + other_value

    return value
