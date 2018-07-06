# -*- coding: utf-8 -*-

from __future__ import unicode_literals

"""
Utility functions
"""

import sys
from six.moves import cPickle
from xml.etree import cElementTree as ET
import logging
from collections import defaultdict
from xml.dom import minidom
import numpy as np
import json
import os
import nltk
import zss

from . import config
from . import datastructures as ds


content_word_tags = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PNOUN'}
UNKNOWN = '<unk>'
EXTRA_EMBEDDINGS = 'extra-embeddings.npy'


class EmbeddingDictionary(object):
    """
    Class for storing a word dictionary to embeddings. It treats special
    cases of OOV words.
    """
    def __init__(self, vocabulary_path, embeddings_path,
                 add_new_vectors=None, path_to_save_extra=None):
        """
        Create a new EmbeddingDictionary.

        :param vocabulary_path: path to vocabulary
        :param embeddings_path: path or paths to 2-d numpy array
        :param add_new_vectors: number of extra embeddings vectors to create, if
            any
        :param path_to_save_extra: path to save the newly created vectors, if
            any
        """
        logging.info('Reading embeddings...')

        # an extra OOV embedding is added to the embeddings. if there was
        # already one and a corresponding <unk> key in wd, the existing one
        # will be used
        self.wd = load_vocabulary(vocabulary_path)
        self.embeddings = load_embeddings(embeddings_path, True,
                                          add_new_vectors, path_to_save_extra)

    def __getitem__(self, item):
        return self.embeddings[self.wd[item]]

    def get_oov_vector(self):
        return self.embeddings[self.wd[UNKNOWN]]

    def set_oov_vector(self, vector):
        self.embeddings[self.wd[UNKNOWN]] = vector

    def get_vectors(self, tokens):
        """
        Return an array with the embeddings of each token in the sentence

        :param tokens: sequence of strings
        :return: numpy array (num_tokens, embedding_size)
        :rtype: np.ndarray
        """
        indices = [self.wd[token] for token in tokens]
        return self.embeddings[indices]


def load_stopwords():
    """
    Return a set of stopwords
    """
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    # add punctuation and the word "é" --- surprisingly, it is lacking
    stopwords.update('.,!?;:"\'()&[]{}-é')
    stopwords.update(['ser', 'estar'])
    return stopwords


def print_cli_args():
    """
    Log the command line arguments
    :return:
    """
    args = ' '.join(sys.argv)
    logging.info('The following command line arguments were given: %s' % args)


def nested_list_to_array(sequences, dtype=np.int32, dim3=None):
    """
    Create a numpy array with the content of sequences.

    In case of sublists with different sizes, the array is padded with zeros.

    If the given sequences are a list of lists, a 2d array is created and dim3
    should be None. In case of a 3-level list, the third level sublists must
    always have the same size and be provided in dim3.

    :param sequences: a list of sublists or 3-level lists. In the latter case,
        the third level sublists must always have the same size.
    :param dtype: type of the numpy array
    :return: a tuple (2d array, 1d array) with the data and the sequence sizes
    """
    if len(sequences) == 0:
        raise ValueError('Empty sequence')

    sizes = np.array([len(seq) for seq in sequences], np.int32)

    if dim3 is None:
        shape = [len(sequences), sizes.max()]
    else:
        shape = [len(sequences), sizes.max(), dim3]

    array = np.zeros(shape, dtype)

    for i, seq in enumerate(sequences):
        if len(seq):
            array[i, :sizes[i]] = seq

    return array, sizes


def load_label_dict(path):
    """
    Load the label dictionary from a json file.

    :param path: either a directory or the file name. In case it is a directory,
        the default file name label-dict.json is used.
    """
    if os.path.isdir(path):
        path = os.path.join(path, 'label-dict.json')

    with open(path, 'r') as f:
        d = json.load(f)

    return d


def write_label_dict(label_dict, path):
    """
    Write a dictionary to a path as a json file
    """
    with open(path, 'w') as f:
        json.dump(label_dict, f)


def load_pickled_pairs(path, add_inverted=False,
                       paraphrase_to_entailment=False):
    """
    Load pickled pairs from the given path.

    :param path: pickle file path
    :param add_inverted: augment the set with the inverted pairs
    :param paraphrase_to_entailment: change paraphrase class to
        entailment
    :return: list of pairs
    """
    logging.info('Reading pairs...')
    with open(path, 'rb') as f:
        pairs = cPickle.load(f)
    logging.info('Read %d pairs' % len(pairs))

    if add_inverted:
        extra_pairs = []
        for pair in pairs:
            if pair.entailment == ds.Entailment.paraphrase:
                ent_value = ds.Entailment.paraphrase
            else:
                # inverting None and Entailment classes yields None
                ent_value = ds.Entailment.none

            inverted = pair.inverted_pair(ent_value)
            extra_pairs.append(inverted)

        pairs.extend(extra_pairs)
        logging.debug('%d total pairs after adding inverted ones' % len(pairs))

    count = 0
    if paraphrase_to_entailment:
        for pair in pairs:
            if pair.entailment == ds.Entailment.paraphrase:
                count += 1
                pair.entailment = ds.Entailment.entailment

        logging.debug('Changed %d paraphrases to entailment' % count)

    return pairs


def load_vocabulary(path):
    """
    Read a file with the vocabulary corresponding to an embedding model.

    :param path: path to the file (should be UTF-8!)
    :return: a python dictionary mapping words to indices in the
        embedding matrix
    """
    with open(path, 'rb') as f:
        text = f.read().decode('utf-8', errors='ignore')

    words = text.splitlines()
    values = range(len(words))
    d = dict(zip(words, values))
    if UNKNOWN not in d:
        d[UNKNOWN] = len(d)

    unk_index = d[UNKNOWN]
    word_dict = defaultdict(lambda: unk_index, d)

    return word_dict


def load_pairs(filename):
    """
    Read the pairs from the given file.

    :param filename: either a TSV or an XML file
    :return: list of pairs
    """
    lower_filename = filename.lower()
    if lower_filename.endswith('.tsv'):
        return load_tsv(filename)
    elif lower_filename.endswith('.xml'):
        return load_xml(filename)
    else:
        msg = 'Unrecognized file extension (expecting .tsv or .xml): %s'
        msg %= filename
        raise ValueError(msg)


def load_tsv(filename):
    """
    Read a TSV file with the format, as used in the SICK corpus.

    sentence1 [TAB] sentence2 [TAB] label [TAB] similarity

    :param filename: path to a tsv file
    :return: list of pairs
    """
    pairs = []
    with open(filename, 'r') as f:
        for line in f:
            sent1, sent2, label = line.strip().split('\t')[:3]
            entailment = map_entailment_string(label)
            pair = ds.Pair(sent1, sent2, None, entailment)
            pairs.append(pair)

    return pairs


def map_entailment_string(ent_string):
    """
    Map an entailment string (written in a corpus file) to an Entailment enum
    object
    """
    ent_string = ent_string.lower()
    if ent_string in ['yes', 'entailment']:
        entailment = ds.Entailment.entailment
    elif ent_string == 'paraphrase':
        entailment = ds.Entailment.paraphrase
    elif ent_string == 'contradiction':
        entailment = ds.Entailment.contradiction
    else:
        entailment = ds.Entailment.none
    return entailment


def load_xml(filename):
    """
    Read an RTE XML file and return a list of Pair objects.
    """
    pairs = []
    tree = ET.parse(filename)
    root = tree.getroot()
    
    for xml_pair in root.iter('pair'):
        t = xml_pair.find('t').text
        h = xml_pair.find('h').text
        attribs = dict(xml_pair.items())
        
        # the entailment relation is expressed differently in some versions
        if 'entailment' in attribs:
            ent_string = attribs['entailment'].lower()
            entailment = map_entailment_string(ent_string)
                        
        elif 'value' in attribs:
            if attribs['value'].lower() == 'true':
                entailment = ds.Entailment.entailment
            else:
                entailment = ds.Entailment.none

        pair = ds.Pair(t, h, None, entailment)
        pairs.append(pair)
    
    return pairs


def write_rte_file(filename, pairs, **attribs):
    """
    Write an XML file containing the given RTE pairs.
    
    :param pairs: list of Pair objects
    """
    root = ET.Element('entailment-corpus')
    for i, pair in enumerate(pairs, 1):
        xml_attribs = {'id':str(i)}
        
        # add any other attributes supplied in the function call or the pair
        xml_attribs.update(attribs)
        xml_attribs.update(pair.attribs)
        
        xml_pair = ET.SubElement(root, 'pair', xml_attribs)
        xml_t = ET.SubElement(xml_pair, 't', pair.t_attribs)
        xml_h = ET.SubElement(xml_pair, 'h', pair.h_attribs)
        xml_t.text = pair.t.strip()
        xml_h.text = pair.h.strip()
        
    # produz XML com formatação legível (pretty print)
    xml_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(xml_string)
    with open(filename, 'wb') as f:
        f.write(reparsed.toprettyxml('    ', '\n', 'utf-8'))


def train_classifier(x, y):
    """
    Train and return a classifier with the supplied data
    """
    classifier = config.classifier_class(class_weight='auto')
    classifier.fit(x, y)
    
    return classifier


def train_regressor(x, y):
    """
    Train and return a regression model (for similarity) with the supplied data.
    """
    regressor = config.regressor_class()
    regressor.fit(x, y)
    
    return regressor


def load_text_embeddings(path):
    """
    Load any embedding model written as text, in the format:
    word[space or tab][values separated by space or tab]

    :param path: path to embeddings file
    :return: a tuple (wordlist, array)
    """
    words = []

    # start from index 1 and reserve 0 for unknown
    vectors = []
    with open(path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            line = line.strip()
            if line == '':
                continue

            fields = line.split(' ')
            word = fields[0]
            words.append(word)
            vector = np.array([float(x) for x in fields[1:]], dtype=np.float32)
            vectors.append(vector)

    embeddings = np.array(vectors, dtype=np.float32)

    return words, embeddings


def load_binary_embeddings(embeddings_path, vocabulary_path):
    """
    Load any embedding model in numpy format, and a corresponding
    vocabulary with one word per line.

    :param embeddings_path: path to embeddings file
    :param vocabulary_path: path to text file with words
    :return: a tuple (wordlist, array)
    """
    vectors = np.load(embeddings_path)

    with open(vocabulary_path, 'rb') as f:
        text = f.read()
    words = [word.decode('utf-8', 'backslashreplace')
             for word in text.splitlines()]

    return words, vectors


def normalize_embeddings(embeddings):
    """
    Normalize the embeddings to have norm 1.

    :param embeddings: 2-d numpy array
    :return: normalized embeddings
    """
    # normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1).reshape((-1, 1))
    return embeddings / norms


def load_embeddings_and_dict(embeddings_path, normalize=True):
    """
    Load and return an embedding model in either text format or
    numpy binary format. If the file extension is .txt, text format is
    assumed. If it is .npy, a corresponding vocabulary file is sought with
    the same name and .txt.

    :param embeddings_path: path to embeddings file
    :param normalize: whether to normalize embeddings
    :return: a tuple (defaultdict, array)
    """
    logging.debug('Loading embeddings')
    base_path, ext = os.path.splitext(embeddings_path)
    if ext.lower() == '.txt':
        wordlist, embeddings = load_text_embeddings(embeddings_path)
    else:
        vocabulary_path = base_path + '.txt'
        wordlist, embeddings = load_binary_embeddings(embeddings_path,
                                                      vocabulary_path)

    wd = {word: ind for ind, word in enumerate(wordlist)}
    if UNKNOWN in wd:
        unk_index = wd[UNKNOWN]
    else:
        unk_index = max(wd.values()) + 1
        wd[UNKNOWN] = unk_index
        mean = embeddings.mean()
        std = embeddings.std()
        shape = [1, embeddings.shape[1]]
        unk_vector = np.random.normal(mean, std, shape)
        embeddings = np.concatenate([embeddings, unk_vector])

    wd = defaultdict(lambda: unk_index, wd)

    logging.debug('Embeddings have shape {}'.format(embeddings.shape))
    if normalize:
        embeddings = normalize_embeddings(embeddings)

    return wd, embeddings


def get_embeddings_path(model_dir):
    """
    Return the path to the extra embeddings vectors

    :param model_dir: path to the directory of the model
    """
    return os.path.join(model_dir, EXTRA_EMBEDDINGS)


def get_vocabulary_path(embeddings_path):
    """
    Return the path of the vocabulary corresponding to the given embeddings

    It just replaces the final .npy with .txt
    """
    return embeddings_path.replace('.npy', '.txt')


def load_embeddings(path_or_paths, normalize=True,
                    add_vectors=None, dir_to_save=None):
    """
    Load an embedding model from the given path

    :param path_or_paths: path or list of paths to a numpy file. If a list,
        all of them will be concatenated in order.
    :param normalize: normalize embeddings to have norm 1
    :param add_vectors: number of vectors to add to the embedding matrix. They
        will be generated randomly with mean and stdev from the others.
    :param dir_to_save: directory to save the newly generated vectors, if any
    :return: numpy array
    """
    if not isinstance(path_or_paths, (list, tuple)):
        path_or_paths = [path_or_paths]

    arrays = []
    for path in path_or_paths:
        arrays.append(np.load(path))

    if add_vectors:
        mean = arrays[0].mean()
        std = arrays[0].std()
        shape = [add_vectors, arrays[0].shape[1]]
        new_vectors = np.random.normal(mean, std, shape)

        path = os.path.join(dir_to_save, EXTRA_EMBEDDINGS)
        np.save(path, new_vectors)
        arrays.append(new_vectors)

    embeddings = np.concatenate(arrays)
    if normalize:
        embeddings = normalize_embeddings(embeddings)

    return embeddings


def create_tedin_dataset(pairs, word_dict, dep_dict, label_dict,
                         lower, use_weights=False):
    """
    Create a Dataset object to feed a Tedin model.

    :param pairs: list of parsed Pair objects
    :param label_dict: dictionary mapping labels to integers
    :param use_weights: whether to use instance weights, inversely proportional
        to the class frequency
    :return: Dataset
    """
    nodes1 = []
    nodes2 = []
    labels = []

    def set_indices(token):
        text = token.text.lower() if lower else token.text
        token.word_index = word_dict[text]
        token.dependency_index = dep_dict[token.dependency_relation]

    for pair in pairs:
        t = pair.annotated_t
        h = pair.annotated_h
        t_indices = []
        h_indices = []

        for token in t.tokens:
            set_indices(token)
            t_indices.append([token.word_index, token.dependency_index])

        for token in h.tokens:
            set_indices(token)
            h_indices.append([token.word_index, token.dependency_index])

        nodes1.append(t_indices)
        nodes2.append(h_indices)
        labels.append(label_dict[pair.entailment.name])

    nodes1, _ = nested_list_to_array(nodes1, dim3=2)
    nodes2, _ = nested_list_to_array(nodes2, dim3=2)
    labels = np.array(labels)

    dataset = ds.Dataset(pairs, nodes1, nodes2, labels=labels,
                         use_weights=use_weights)

    return dataset


def split_positive_negative(pairs):
    """
    Split a list of pairs into two lists: one containing only positive
    and the other containing only negative pairs.

    :return: tuple (positives, negatives)
    """
    positive = [pair for pair in pairs
                if pair.entailment == ds.Entailment.entailment
                or pair.entailment == ds.Entailment.paraphrase]
    neutrals = [pair for pair in pairs
                if pair.entailment == ds.Entailment.none]

    return positive, neutrals


def load_positive_and_negative_data(path, word_dict, dep_dict, label_dict=None,
                                    lower=True):
    """
    Load a pickle file with pairs and do some necessary preprocessing for the
    pairwise ranker.

    :param path: path to saved pairs in pickle format
    :param word_dict: dictionary mapping words to integers
    :param dep_dict: dictionary mapping dependency labels to integers
    :param label_dict: dictionary mapping RTE labels to integers or None
        if None, one will be created
    :param lower: whether to lowercase tokens
    :return: tuple of ds.Datasets (positive, negative)
    """
    pairs = load_pickled_pairs(path)
    if label_dict is None:
        label_dict = create_label_dict(pairs)

    pos_pairs, neg_pairs = split_positive_negative(pairs)
    pos_data = create_tedin_dataset(pos_pairs, word_dict, dep_dict, label_dict,
                                    lower)
    neg_data = create_tedin_dataset(neg_pairs, word_dict, dep_dict, label_dict,
                                    lower)

    msg = '%d positive and %d negative pairs' % (len(pos_data), len(neg_data))
    logging.info(msg)

    return pos_data, neg_data


def create_label_dict(pairs):
    """
    Create a dictionary mapping label names (entailment, none, etc) to integers
    """
    labels = set(pair.entailment.name for pair in pairs)
    label_dict = {label: i for i, label in enumerate(labels)}
    return label_dict


def create_dependency_dict(pairs):
    """
    Create a dictionary mapping dependency labels (nsubj, dobj, etc) to integers
    """
    dep_set = set()
    for pair in pairs:
        sent1 = pair.annotated_t
        sent2 = pair.annotated_h
        deps1 = [token.dependency_relation for token in sent1.tokens]
        deps2 = [token.dependency_relation for token in sent2.tokens]
        dep_set.update(deps1)
        dep_set.update(deps2)

    dep_dict = {dep: i for i, dep in enumerate(dep_set)}
    return dep_dict


def extract_classes(pairs, label_dict=None):
    """
    Extract the class infomartion (paraphrase, entailment, none, contradiction)
    from the pairs.

    :param pairs: list of pairs
    :param label_dict: if given, must be a mapping from entailment strings to
        ints. If not given, one will be generated
    :return: if label_dict is given, a numpy array with values from 0 to
        num_classes - 1.
        If label_dict is not given, a tuple with the array and a label_dict
    """
    if label_dict is None:
        label_dict = create_label_dict(pairs)

    classes = np.array([label_dict[pair.entailment.name] for pair in pairs])

    return classes, label_dict


def load_tedin_data(path, word_dict, dep_dict, label_dict=None, lower=True,
                    use_weights=False):
    """
    Load a pickle file with pairs and return a dataset for the tedin model.

    :param path: path to pickle file
    :param word_dict: dictionary mapping tokens to integers
    :param dep_dict: dictionary mapping dependency labels to integers
    :param lower: whether to lowercase tokens before looking in the word_dict
    :param label_dict: if given, must be a mapping from entailment strings to
        ints. If not given, one will be generated
    :param use_weights: use class weights to counter class unbalance
    :return: dataset, label_dict
    """
    pairs = load_pickled_pairs(path)

    if label_dict is None:
        label_dict = create_label_dict(pairs)

    data = create_tedin_dataset(pairs, word_dict, dep_dict, label_dict,
                                lower, use_weights)
    return data, label_dict


def get_logger(name='logger'):
    """
    Setup and return a simple logger.
    :return:
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    logger.propagate = False

    return logger


def run_zss(sent1, sent2):
    """
    Compute a simple tree edit distance (TED) value and operations.

    Nodes are considered to match if they have the same dependency label and
    lemma.
    """
    def get_children(node):
        return node.dependents

    def get_label(node):
        return node.lemma, node.dependency_relation

    def label_dist(label1, label2):
        return int(label1 != label2)

    root_t = sent1.root
    root_h = sent2.root

    return zss.simple_distance(root_t, root_h, get_children, get_label,
                               label_dist, True)
