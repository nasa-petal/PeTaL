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
