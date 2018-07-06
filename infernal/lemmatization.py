# -*- coding: utf-8 -*-

from __future__ import unicode_literals

'''
Functions for dealing with lemmatization.
'''

import logging

from . import config

unitex_dictionary = None
udtags2delaf = {'VERB': 'V',
                'AUX': 'V',
                'NOUN': 'N',
                'ADJ': 'A'}
lemmatizable_delaf_tags = set(udtags2delaf.values())
lemmatizable_ud_tags = set(udtags2delaf.keys())


def _load_unitex_dictionary():
    '''
    Read the Unitex dictionary with inflected word forms. 
    
    This function ignores all entries with clitics because it
    is targeted at a system that uses Universal Dependencies and
    treats verbs and clitics separately, and removing clitics 
    saves memory. 
    '''
    global unitex_dictionary
    
    logging.info('Reading Unitex dictionary')
    with open(config.unitex_dictionary_path, 'rb') as f:
        unitex_dictionary = {}
        for line in f:
            line = line.decode('utf-8').strip()
            # each line is in the format
            # inflected_word,lemma.POS:additional_morphological_metadata
            # the morphological metadata is only available for open class words
            inflected, rest = line.split(',')
            if '-' in inflected:
                continue
            
            lemma, morph = rest.split('.')
            if ':' in morph:
                pos, _ = morph.split(':')
            else:
                pos = morph
            
            if pos not in lemmatizable_delaf_tags:
                continue

            unitex_dictionary[(inflected, pos)] = lemma
            
    logging.info('Finished')


def get_lemma(word, pos):
    '''
    Retrieve the lemma of a word given its POS tag.
    
    If the combination of word and POS is not known, return the
    word itself. 
    '''
    global unitex_dictionary
    if unitex_dictionary is None:
        _load_unitex_dictionary()
    
    if '\0' in word:
        logging.error('\\0 in {}'.format(word))
    
    word = word.lower()
    if pos not in lemmatizable_ud_tags:
        return word
    delaf_pos = udtags2delaf[pos]
    
    if (word, delaf_pos) not in unitex_dictionary:
        # a lot of times, this happens with signs like $ or %
        if len(word) == 1:
            return word

        # UNITEX has a Brazilian vocabulary; some sequences are typical from
        # European Portuguese
        retry = False
        if 'ct' in word:
            word = word.replace('ct', 't')  # facto -> fato
            retry = True
        elif 'cç' in word:
            word = word.replace('cç', 'ç')  # acção -> ação
            retry = True
        elif 'pt' in word:
            word = word.replace('pt', 't')  # óptimo -> ótimo
            retry = True
        elif 'pç' in word:
            word = word.replace('pç', 'ç')  # adopção -> adoção
            retry = True
        elif 'oo' in word:
            word = word.replace('oo', 'ôo')  # voo -> vôo
            retry = True
        elif 'ee' in word:
            word = word.replace('ee', 'êe')
            retry = True
        elif 'mn' in word:
            word = word.replace('mn', 'n')
            retry = True

        if retry and (word, delaf_pos) in unitex_dictionary:
            return unitex_dictionary[(word, delaf_pos)]

        # the POS tag could be wrong
        # but nouns and adjectives are more likely to be mistaken for each other
        #TODO: check if it is a good idea to allow changes from noun/adj to verb
        if delaf_pos == 'N':
            try_these = ['A', 'V']
        elif delaf_pos == 'A':
            try_these = ['N', 'V']
        else:
            try_these = ['N', 'A']
        
        for other_pos in try_these:            
            if (word, other_pos) in unitex_dictionary:
                logging.debug('Could not find lemma for word {} with POS {},'\
                              'but found for POS {}'.format(word, delaf_pos, other_pos))
                return unitex_dictionary[(word, other_pos)]
        
        msg = 'Could not find lemma for word {} (tagged {}, but tried other tags)'
        logging.info(msg.format(word, delaf_pos))
        return word
        
    return unitex_dictionary[(word, delaf_pos)]
