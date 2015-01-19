#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import multiprocessing
import re
import sys
import xml.etree.ElementTree as ET
from HTMLParser import HTMLParser


class Writer(multiprocessing.Process):
    """writes stuff"""
    def __init__(self, oq, filename):
        multiprocessing.Process.__init__(self)
        self.oq = oq
        self.filename = filename

    def run(self):
        f = open(self.filename, 'w')
        while True:
            line = self.oq.get()
            if line is None:
                break
            f.write(line)
        f.close()
        return


class Worker(multiprocessing.Process):
    """works"""
    def __init__(self, iq, oq):
        multiprocessing.Process.__init__(self)
        self.iq = iq
        self.oq = oq

    def run(self):
        while True:
            doc = self.iq.get()
            if doc is None:
                break
            out = self.parse_rec(doc.strip())
            if out is not None:
                self.oq.put(out)
        return

    class MLStripper(HTMLParser):
        def __init__(self):
            self.reset()
            self.fed = []

        def handle_data(self, d):
            self.fed.append(d)

        def get_data(self):
            return ''.join(self.fed)

    def strip_stuff(self, html):
        s = self.MLStripper()
        s.feed(html)
        return re.sub('[\t\n\r]', ' ', s.unescape(s.get_data()))

    def parse_rec(self, xmlstring):
        try:
            root = ET.fromstring(xmlstring)
        except:
            return None
        abstract_text = ""
        abstract = root.find("./MedlineCitation/Article/Abstract")
        if abstract is not None:
            for entry in [t for t in abstract.findall("AbstractText")
                          if t.text is not None]:
                abstract_text += entry.text + " "
            PMID = root.find("./MedlineCitation/PMID").text
            return PMID + "\t" + (self
                                  .strip_stuff(abstract_text)
                                  .encode("ascii", "ignore")) + "\n"
        return None


if __name__ == '__main__':
    cpus = multiprocessing.cpu_count()
    parser = argparse.ArgumentParser(description="Extract abstracts from xml")
    parser.add_argument('xml_file', help="The XML file to parse")
    parser.add_argument('csv_file', help="The output file name")
    parser.add_argument('-p', '--processes', type=int, default=(cpus * 2) - 1,
                        help='number of processors')

    args = parser.parse_args()

    xml_file = args.xml_file
    csv_file = args.csv_file
    processes = args.processes

    iq = multiprocessing.Queue(1000)
    oq = multiprocessing.Queue()

    writer = Writer(oq, csv_file)
    writer.start()
    workers = [Worker(iq, oq) for i in xrange(processes)]
    for worker in workers:
        worker.start()

    with open(xml_file) as xmlfile:
        print("Parsing...\n")
        for num, line in enumerate(xmlfile):
            iq.put(line)
            if num % 1000 == 0:
                sys.stdout.write("\rRecord: %d" % num)
                sys.stdout.flush()

        for i in xrange(processes):
            iq.put(None)

        for worker in workers:
            worker.join()

        oq.put(None)
        writer.join()
