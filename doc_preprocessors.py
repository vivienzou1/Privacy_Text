'''
loading pre-process corpus of documents (tsv)
'''
from snorkel.parser import TSVDocPreprocessor
doc_preprocessor = TSVDocPreprocessor('dataset.tsv', max_docs=n_docs)