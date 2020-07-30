from petal_site import settings
from neo4j import GraphDatabase, basic_auth
from pandas import DataFrame

from .cleaner import Cleaner

from pprint import pprint
import pickle
from time import time

def clean(item):
    if item is None:
        return None
    item = str(item)
    item = item.replace('-', '_')
    item = item.replace('\\', '_')
    item = item.replace('/', '_')
    item = item.replace('\'', '')
    item = item.replace('(', '')
    item = item.replace(')', '')
    return item

neo4j_settings = settings.NEO4J_DATABASE

# Load index and db connection at import
neo_client = GraphDatabase.driver(neo4j_settings['url'], auth=basic_auth(neo4j_settings['username'], neo4j_settings['password']), encrypted=False)
session = neo_client.session()

try:
    with open('./data/index', 'rb') as infile:
        index = pickle.load(infile)
        infile.close()
except IOError:
    index = []

def fetch(query):
    cleaner = Cleaner()
    articles = set()

    start = time()
    result_lists = []
    for word in cleaner.clean(query):
        if word in index:
            result_lists.append(index[word])
    finish = time()
    for results in result_lists:
        for *hits, uuid in results:
            result = session.run('MATCH (a:Article) WHERE a.uuid = \'{uuid}\' RETURN a'.format(uuid=clean(uuid)))
            for article in (node['a'] for node in result):
                article.__hash__ = lambda x : x['title']
                articles.add(article)
    fetch_finish = time()
    return finish - start, fetch_finish - finish, articles

def search(query):
    query_time, fetch_time, articles = fetch(query)
    articles = [(dict(title=a['title'], abstract=a['abstract'], url=a['url'])) for a in articles] 
    context = dict(search_time=round(query_time, 10), neo4j_time=round(fetch_time, 10), articles=articles)
    return context

def biomole_search(query):
    query_time, fetch_time, articles = fetch(query)
    articles = [(dict(title=a['title'], url=a['url'])) for a in articles] 
    context = dict(articles=articles)
    return context

import plotly.express as px
import plotly.offline as opy
import plotly.graph_objs as go

def plot(query):
    _, _, articles = fetch(query)

    habitats = []
    for article in articles:
        taxa = session.run('MATCH (t:Taxon)-->(a:Article) WHERE a.uuid = \'{uuid}\' RETURN t'.format(uuid=article['uuid']))
        for taxon in (node['t'] for node in taxa):
            query_habitats = session.run('MATCH (t:Taxon)-[:habitat]->(h) WHERE t.name = \'{name}\' return h'.format(name=taxon['name']))
            for node in query_habitats:
                habitats.append(node['h']['value'])

    pprint(habitats)
    mock_df = DataFrame(dict(habitats=habitats))
    fig = px.histogram(mock_df, x='habitats', nbins=len(set(habitats)), title='Habitats Related To Search Query', template='plotly_dark')
    return dict(graph=opy.plot(fig, auto_open=False, output_type='div'))

def main():
    search('megaptera')

if __name__ == '__main__':
    main()
