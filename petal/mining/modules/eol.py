from .eol_api import EOL_API
from .module import Module

from pprint import pprint

from bs4 import BeautifulSoup
from requests import get
from time import sleep

from pprint import pprint

import sys, re, os

search_url = 'https://eol.org/search?utf8=%E2%9C%93&q={}'
data_url = 'https://eol.org{}/data'
eol_url = 'https://eol.org{}'

def get_data_page(query):
    expanded = '+'.join(query.split()) + '+'
    url = search_url.format(expanded)

    html = get(url).text
    processed = BeautifulSoup(html, 'html.parser')
    top = processed.find(attrs={'class' : 'search-result'})
    link = top.find('a')
    nav = link.get('href')
    return data_url.format(nav)


def read_data(page_url):
    extracted = []

    data = get(page_url)
    processed = BeautifulSoup(data.text, 'html.parser')
    rows = processed.find_all(attrs={'class', 'js-data-row'})
    for row in rows:
        source = row.find(attrs={'class', 'trait-source'})
        link = source.find('a')
        nav  = eol_url.format(link.get('href'))
        source_text = link.text
        data_div = row.find(attrs={'class', 'trait-data'})
        subdivs  = data_div.find_all('div')
        header = subdivs[0].find('div').text.strip()
        entry  = subdivs[1].text.strip()
        row_p = (nav, source_text, header, entry)
        extracted.append(row_p)
    return extracted

def search(query):
    page_url = get_data_page(query)
    return read_data(page_url)

class EOLModule(Module):
    def __init__(self, in_label='Species', out_label='EOLData', connect_labels=('MENTIONED_IN_DATA', 'MENTIONS_SPECIES')):
        Module.__init__(self, in_label, out_label, connect_labels)
        self.api = EOL_API()

    def process(self, node):
        print(node)
        name = node['name']
        # pprint(self.api.search('MATCH (p:Page)-[:trait]->(t:Trait)-[:metadata]->(m) WHERE p.canonical=\'{name}\' RETURN m LIMIT 100'.format(name=name)))
        query = 'MATCH (t:Trait)<-[:trait]-(p:Page),(t)-[:supplier]->(r:Resource),(t)-[:predicate]->(pred:Term) WHERE p.canonical=\'{name}\' OPTIONAL MATCH (t)-[:object_term]->(obj:Term) OPTIONAL MATCH (t)-[:normal_units_term]->(units:Term) OPTIONAL MATCH (lit:Term) WHERE lit.uri = t.literal RETURN r.resource_id, t.eol_pk, t.resource_ok, t.source, p.page_id, t.scientific_name, pred.uri, pred.name, t.object_page_id, obj.uri, obj.name, t.normal_measurement, units.uri, units.name, t.normal_units, t.literal, lit.name LIMIT 5'.format(name=name)
        pprint(self.api.search(query))
        # pprint(self.api.search('MATCH (p:Page)-[x:trait]->(t:Trait)-[:metadata]->(m) WHERE p.canonical=\'{name}\' RETURN x LIMIT 100'.format(name=name)))
        # pprint(self.api.search('MATCH (p:Page) WHERE p.canonical=\'{name}\' RETURN p.trait LIMIT 100'.format(name=name)))
        # pprint(search(name))
        1/0
        return dict()

