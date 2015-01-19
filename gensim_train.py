#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import multiprocessing
import sys
from gensim import utils
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer


class Sentences(object):
    """Yields LabeledSentences"""
    def __init__(self, oq):
        self.oq = oq

    def __iter__(self):
        while True:
            line = self.oq.get()
            if line is None:
                break
            yield line
        return


class Preprocessor(multiprocessing.Process):
    """works"""
    def __init__(self, iq, oq):
        multiprocessing.Process.__init__(self)
        self.iq = iq
        self.oq = oq
        self.sent_tokenizer = PunktSentenceTokenizer()
        self.re_tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = LancasterStemmer()

    def run(self):
        while True:
            doc = self.iq.get()
            if doc is None:
                break
            out = self.process(doc)
            if out is not None:
                self.oq.put(out)
        return

    def process(self, line):
            line = utils.to_unicode(line)
            fields = line.strip().split("\t")
            PMID = ["PMID" + fields[0]]
            sentences = self.sent_tokenizer.tokenize(fields[0].lower())
            for sent in sentences:
                words = [self.stemmer.stem(w) for w
                         in self.re_tokenizer.tokenize(sent)]
                return LabeledSentence(words, PMID)


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

    iq = multiprocessing.Queue(1000)
    oq = multiprocessing.Queue()

    sentences = Sentences(oq)
    sentences.start()
    workers = [Preprocessor(iq, oq) for i in xrange(processes)]
    for worker in workers:
        worker.start()

    with open(csv_file) as csvfile:
        print("Building vocab...\n")
        for num, line in enumerate(csvfile):
            iq.put(line)
            if num % 1000 == 0:
                sys.stdout.write("\rRecord: %d" % num)
                sys.stdout.flush()

        for i in xrange(processes):
            iq.put(None)

        for worker in workers:
            worker.join()

        oq.put(None)
        Preprocessor.join()

    model = Doc2Vec(size=dimension, window=window,
                    min_count=min_count, workers=processes)
    model.build_vocab(sentences)

    sentences = Sentences(oq)
    sentences.start()
    workers = [Preprocessor(iq, oq) for i in xrange(processes)]
    for worker in workers:
        worker.start()

    with open(csv_file) as csvfile:
        print("\nTraining...\n")
        for num, line in enumerate(csvfile):
            iq.put(line)
            if num % 1000 == 0:
                sys.stdout.write("\rRecord: %d" % num)
                sys.stdout.flush()

        for i in xrange(processes):
            iq.put(None)

        for worker in workers:
            worker.join()

        oq.put(None)
        Preprocessor.join()

    model.train(sentences)
    model.save(model_file)
