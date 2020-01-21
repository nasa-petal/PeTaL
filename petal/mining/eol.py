from bs4 import BeautifulSoup
from requests import get
from time import sleep

from pprint import pprint

import sys, re, os

search_url = 'https://eol.org/search?utf8=%E2%9C%93&q={}'
data_url = 'https://eol.org{}/data'

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
    data = get(page_url)
    processed = BeautifulSoup(data.text, 'html.parser')
    rows = processed.find_all(attrs={'class', 'js-data-row'})
    print('Found ', len(rows), ' results')


def main():
    example = ('Animalia', 'Acanthocephala', 'Archiacanthocephala', 'Apororhynchida', 'Apororhynchidae', 'Apororhynchus', 'Apororhynchus aculeatus')
    print('Downloading data from EOL')
    species = example[-1]

    page_url = get_data_page(species)
    read_data(page_url)

if __name__ == '__main__':
    main()
