# -*- coding: utf-8 -*-

from __future__ import unicode_literals

'''
This is the global configuration file. It contains configurations for resources
and external tools that are shared by all configurations.
'''

# ==============
# Parsing config
# ==============

corenlp_url = 'http://localhost'
corenlp_port = 9000

# path to the corenlp models inside the server
corenlp_depparse_path = r'models/pt-br/dep-parser'
corenlp_pos_path = 'models/pt-br/pos-tagger.dat'

# label of the dependency relation indicating negation 
negation_rel = 'neg'


# ========================
# Lexical resources config
# ========================

stopwords_path = None

# pickle is faster to read than wordnet in nt or ppdb in txt
ownpt_path = 'data/own-pt.pickle'
ppdb_path = 'data/ppdb-xxl-phrasal.pickle'

unitex_dictionary_path = 'data/Delaf2015v04.dic'
