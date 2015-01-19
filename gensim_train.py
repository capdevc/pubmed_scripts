#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import multiprocessing
from gensim import utils
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer



class DocReader(object):
    """Generator to iterate through a file, providing sentences"""
    def __init__(self, filename):
        self.filename = filename
        self.sent_tokenizer = PunktSentenceTokenizer()
        self.re_tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = LancasterStemmer()

    def __iter__(self):
        with open(self.filename) as in_file:
            for line in in_file:
                line = utils.to_unicode(line)
                fields = line.strip().split("\t")
                PMID = ["PMID" + fields[0]]
                sentences = self.sent_tokenizer.tokenize(fields[0].lower())
                for sent in sentences:
                    words = [self.stemmer.stem(w) for w
                             in self.re_tokenizer.tokenize(sent)]
                    yield LabeledSentence(words, PMID)

if __name__ == '__main__':
    cpus = multiprocessing.cpu_count()
    parser = argparse.ArgumentParser(description="Train a doc2vec model")
    parser.add_argument('csv_file', help="The tab delimited abstract csv file")
    parser.add_argument('model_file', help="The output file name")
    parser.add_argument('-d', '--dimension', type=int, default=1600,
                        help="Dimension of vectors")
    parser.add_argument('-w', '--window', type=int, default=8,
                        help="Window size")
    parser.add_argument('-m', '--mincount', type=int, default=5,
                        help="Minimum word count")
    parser.add_argument('-p', '--processes', type=int, default=cpus,
                        help='Number of worker processes')

    args = parser.parse_args()

    csv_file = args.csv_file
    model_file = args.model_file
    dimension = args.dimension
    window = args.window
    min_count = args.mincount
    processes = args.processes

    dr = DocReader(csv_file)
    model = Doc2Vec(sentences=dr, size=dimension, window=window,
                    min_count=min_count, workers=processes)

    model.save(model_file)

