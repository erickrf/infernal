# -*- coding: utf-8 -*-

'''
Functions to read the OpenWordnetPT from RDF files and provide
access to it.
'''

import rdflib
from six.moves import cPickle


ownns = rdflib.Namespace('https://w3id.org/own-pt/wn30/schema/')
nomlexns = rdflib.Namespace('https://w3id.org/own-pt/nomlex/schema/')
lexical_form_predicate = rdflib.URIRef(
    u'https://w3id.org/own-pt/wn30/schema/lexicalForm')
word_type = ownns['Word']
word_pred = ownns['word']
word_sense_type = ownns['WordSense']
type_pred = rdflib.RDF.type
contains_sense_pred = ownns['containsWordSense']
nomlex_verb_pred = nomlexns['verb']
nomlex_noun_pred = nomlexns['noun']

_wn_graph = None


def load_wordnet(path, force=False):
    """
    Load the wordnet graph from the given path. A call to this function
    is necessary before using the other ones in this module.

    :param path: path to either a .pickle or .nt file. If it is a pickled 
        file, it should contain a previously serialized wordnet graph.
    :param force: if True, reloads the file even if one has been previously
        loaded.
    """
    global _wn_graph
    if _wn_graph is not None and not force:
        return

    if path.endswith('.nt'):
        _wn_graph = rdflib.Graph()
        _wn_graph.parse(path, format='nt')
    elif path.endswith('.pickle'):
        with open(path, 'rb') as f:
            _wn_graph = cPickle.load(f)
    else:
        raise ValueError('Wordnet file extension is neither .nt or .pickle')


def find_synonyms(word):
    """
    Find all synonyms of the given word in the wordnet graph, considering
    all possible synsets.

    :return: a set of unicode strings
    """
    synonyms = set()
    synsets = find_synsets(word)
    for synset in synsets:
        synonyms.update(get_synset_words(synset))

    return synonyms


def are_synonyms(word1, word2):
    """
    Return True if word1 and word2 share at least one synset in graph.
    """
    synsets1 = find_synsets(word1)
    synsets2 = find_synsets(word2)
    return len(synsets1.intersection(synsets2)) > 0


def get_word_node(word):
    """
    Return the RDF node used in own-pt to represent a given word.
    """
    word_literal = rdflib.Literal(word, 'pt')
    word_node = _wn_graph.value(None, lexical_form_predicate, word_literal)
    return word_node


def word_node_to_string(word_node):
    """
    Return the string corresponding to the given own-pt word node.
    """
    word_literal = _wn_graph.value(word_node, lexical_form_predicate,
                                   any=False)
    return word_literal.toPython()


def find_synsets(word):
    '''
    Find and return all synsets containing the given word in the given graph.

    :param word: unicode string
    :return: a set of synsets (rdflib objects). It is empty is the word is not
        in the wordnet
    '''
    all_synsets = set()
    word_node = get_word_node(word)

    if word_node is None:
        # this word is not in the wordnet
        return all_synsets

    # word nodes are linked to word sense nodes
    word_senses_iter = _wn_graph.subjects(word_pred, word_node)

    for word_sense in word_senses_iter:
        synsets_iter = _wn_graph.subjects(contains_sense_pred, word_sense)
        synsets = list(synsets_iter)
        all_synsets.update(synsets)

    return all_synsets


def get_synset_words(synset):
    '''
    Return the words of a synset

    :return: a list of strings
    '''
    words = []

    # a synset has many word senses
    # each word sense has a Word object and each Word has a lexical form
    senses = _wn_graph.objects(synset, contains_sense_pred)
    for sense in senses:
        word_node = _wn_graph.value(sense, word_pred, any=False)
        words.append(word_node_to_string(word_node))

    return words


def find_nominalizations(word):
    """
    Find and return nominalizations of the given verb.

    :return: a list of possible nominalizations, as strings
    """
    word_node = get_word_node(word)
    if word_node is None:
        return []

    nouns = []
    # a nominalization object links nouns and verbs
    nominalizations = _wn_graph.subjects(nomlex_verb_pred, word_node)
    for nom in nominalizations:
        noun = _wn_graph.value(nom, nomlex_noun_pred, None)
        noun_string = word_node_to_string(noun)
        nouns.append(noun_string)

    return nouns
