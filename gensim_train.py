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


class DocReader(multiprocessing.Process):
    """read lines from file and queue"""
    def __init__(self, iq, oq, processes, filename):
        multiprocessing.Process.__init__(self)
        self.iq = iq
        self.oq = oq
        self.processes = processes
        self.filename = filename

    def __iter__(self):
        while True:
            line = self.oq.get()
            if line is None:
                break
            yield line
        return

    def run(self):
        self.processors = [Preprocessor(self.iq, self.oq)
                           for x in xrange(self.processes)]
        for processor in self.processors:
            processor.start()
        with open(self.filename) as infile:
            for num, line in enumerate(infile):
                self.iq.put(line)
                if num % 1000 == 0:
                    sys.stdout.write("\rRecord: %d" % num)
                    sys.stdout.flush()
        for processor in self.processors:
            self.iq.put(None)
        for i, processor in enumerate(self.processors):
            processor.join()

        self.oq.put(None)
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
            if len(fields) < 2:
                return None
            PMID = ["PMID" + fields[0]]
            sentences = self.sent_tokenizer.tokenize(fields[1].lower())
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
    model = Doc2Vec(size=dimension, window=window,
                    min_count=min_count, workers=processes)

    print("Building vocab...")
    dr = DocReader(iq, oq, processes, csv_file)
    dr.start()
    model = Doc2Vec(size=dimension, window=window,
                    min_count=min_count, workers=processes)
    model.build_vocab(dr)

    print("\nTraining...")
    dr = DocReader(iq, oq, processes, csv_file)
    dr.start()
    model.train(dr)
    model.save(model_file)
