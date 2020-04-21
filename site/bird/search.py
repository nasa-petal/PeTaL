from modules.libraries.natural_language.cleaner import Cleaner

from neo4j import GraphDatabase, basic_auth

from pprint import pprint
import pickle
from time import time

# Load index and db connection at import
neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"), encrypted=False)
session = neo_client.session()
with open('../pipeline/data/index', 'rb') as infile:
    index = pickle.load(infile)

def fetch(query):
    cleaner = Cleaner()
    articles = []
    for term in cleaner.clean(query):
        if term in index:
            term_results = index[term]
            results.append(term_results)
            for a, b, uuid in term_results:
                article = session.run('MATCH (a:Article) WHERE a.uuid = \'{uuid}\' RETURN a'.format(uuid=uuid))
                article = next(article.records())['a']
                articles.append(article)
    return articles

def search(query):
    start = time()
    articles = fetch(query)
    articles = [(dict(title=a['title'], abstract=a['summary'], url=a['url'])) for a in articles] 
    done = time()
    context = dict(search_time=round(done - start, 10), articles=articles)
    return context

import plotly.express as px
import plotly.offline as opy
import plotly.graph_objs as go

def plot(query):
    search_context = search(query)
    articles = search_context['articles']

    fig = px.histogram(df, x='x', nbins=10)
    div = opy.plot(fig, auto_open=False, output_type='div')
    context['graph'] = div
    return context

def main():
    search('megaptera')

if __name__ == '__main__':
    main()
