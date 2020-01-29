import wikipedia

from pprint import pprint
from collections import namedtuple

import sys
import codecs
import random

from topic_modeler import TopicModeler

def set_unicode_io():
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)
    sys.stderr = codecs.getwriter('utf8')(sys.stderr)

Node = namedtuple('Node', ['name', 'path', 'dist'])
Node.__hash__ = lambda self: hash(self.name)
Node.__new__.__defaults__ = (None,) * 5

PAGE_DAMPING = 0.85
PAGE_DAMPING = 1.0

def check_damping():
    return random.random() < PAGE_DAMPING

def main():
    topic_finder = TopicModeler()
    pages = []
    for i, link in enumerate(wikipedia.page('Biology').links):
        try:
            pages.append(wikipedia.page(link).summary)
            print(link, flush=True)
        except wikipedia.exceptions.DisambiguationError:
            pass
        if i == 100:
            break
    topic_finder.update(pages)
    topic_finder.print_topics()

    # relate(a='Aerodynamics', b='Bird flight')
    # relate(a='Engineering', b='Biology')

def expand(nodes, meta=None):
    if meta is None:
        meta = dict()
    next_set = set()
    for node in nodes:
        if check_damping():
            page = wikipedia.page(node.name)
            for link in page.links:
                next_set.add(Node(name=link, path=node.path + (link,), dist=node.dist + 1))
                meta['count'] += 1
                print('{}\r'.format(meta['count']), flush=True, end='')
    print(meta['count'])
    return next_set

def relate(a='Biology', b='Engineering'):
    bio = {Node(name=a,     dist=0, path=(a,))}
    eng = {Node(name=b, dist=0, path=(b,))}
    meta = dict(count=2)

    for i in range(3):
        bio = expand(bio, meta=meta)
        eng = expand(eng, meta=meta)
        overlap = set.intersection(bio, eng)
        print(overlap)

# set_unicode_io()
if __name__ == '__main__':
    main()
