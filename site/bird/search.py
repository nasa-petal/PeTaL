from modules.libraries.natural_language.cleaner import Cleaner

from neo4j import GraphDatabase, basic_auth
from pandas import DataFrame

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

    mock_df = DataFrame(dict(x=['aquatic'] * 3 + ['terrestrial'] * 10 + ['airborne'] * 6))
    fig = px.histogram(mock_df, x='x', nbins=3, template='plotly_dark')
    return dict(graph=opy.plot(fig, auto_open=False, output_type='div'))

def main():
    search('megaptera')

if __name__ == '__main__':
    main()
