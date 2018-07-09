Infernal
========

Infernal (INFERence in NAtural Language) is a model for performing natural
language inference / recognizing textual entailment based on handcrafted
features. It was implemented primarily for Portuguese, but most of it can be
reused for other languages.

Reference
---------

If you publish research using or expanding on Infernal, please cite:

* Erick Fonseca and Sandra M. Aluísio. Syntactic Knowledge for Natural Language
Inference in Portuguese. In: Proceedings of the 2018 International Conference
on the Computational Processing of Portuguese (PROPOR). 2018.
*(accepted for publication)*

```
@inproceedings{infernal,
  author = {Erick Fonseca and Sandra M. Alu\'isio},
  title = {{Syntactic Knowledge for Natural Language Inference in Portuguese}},
  year = {2018},
  booktitle = {Proceedings of the 2018 International Conference
on the Computational Processing of Portuguese (PROPOR)}
}
```

Requirements
------------

In order to run the full pre-processing pipeline, you will need (besides the
libraries in `requirements.txt`, the following:

* A working installation of [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/), not necessarily in the same machine running Infernal.
 You will need trained models for parsing and POS tagging in Portuguese; these are available [here](https://docs.google.com/uc?export=download&id=0B9AWPiAx5i1lb3ZLWVVCTkJBSTQ).

* The [DELAF dictionary](http://www.nilc.icmc.usp.br/nilc/projects/unitex-pb/web/dicionarios.html).
 By default, Infernal expects the dictionary file `Delaf2015v04.dic` to be in a directory `data` under the infernal root.

* The spaCy Portuguese model. Download it with `python -m spacy download pt`.

* [OpenWordNet-PT](https://github.com/own-pt/openWordnet-PT). Copy the `own-pt.nt` (originally gzipped) file to the `data` directory under infernal root.

Configuration
-------------

The `config.py` file (under the directory `infernal`) has some filenames and endpoints to be configured.
You should change file names to match how you saved the required files above.

Also, configure properly the CoreNLP URL and port. The path to POS tagger and dependency tagger are directories
inside the CoreNLP root folder (again, it might be in the same machine or not).

Then, install the module (maybe with `--user` if you're in a shared environment and not using virtualenv):

```
python setup.py install
```

Preprocessing
-------------

### Save OpenWordNet-PT as .pickle

First, it is a good idea to convert the original OpenWordNet-PT file from NT to a pre-processed pickle, which is much faster
to read. NT is a generic data format, while the pickle has everything in the format used by Infernal.

```
python scripts/serialize-wordnet.py data/own-pt.nt data/own-pt.pickle
```

### Pre-process the pairs

Next, take the raw pairs and tokenize, parse, find lemmas and named entities.
Make sure the CoreNLP endpoint is running and run:

```
python scripts/preprocess.py pairs.xml pairs.pickle
```

You should repeat this process for training, validation and test data.

### Extract features

Now that you have nice parsed pairs, you can compute the features for them.
After computing features (which take some time) and saving them, you can try
different classifiers without repeating the feature extraction.

```
python scripts/extract-features.py pairs.pickle word-embeddings.npy features.npy [--load-label-dict DICT] [--save-label-dict DICT]
```

The label dict is a simple dictionary converting labels (*entailment*,
*neutral*, *paraphrase*) to number codes. The first time you run
`extract-features.py`, save the label dictionary to your data directory. Then,
when you run it again for validation and test data, load the dict and the labels
will get the same codes.

The word embeddings should be a 2-d array saved in the numpy `.npz` format, and
have shape `(vocabulary_size, embedding_size)`. Additionally, a file in the same
directory in `.txt` format must have the embedding vocabulary.

Training the model
------------------

Finally, train a model:

```
python scripts/train-infernal.py features.npy log-regression model/ -s
```

Run `python scripts/train-infernal.py` to see the available options. Since
choosing the right algorithm and tuning the model can be quite complex, maybe
you'll want to change something in the training code.

Evaluating the model
--------------------

To evaluate model performance:

```
python scripts/evaluate.py features.npy model/
```

The `features.npy` (or whatever you generated with `extract-features.py`)
has both features and classes, and is faster to read than parsing an XML file.
