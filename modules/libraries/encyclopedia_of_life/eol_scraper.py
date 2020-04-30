from pprint import pprint

from bs4 import BeautifulSoup
from requests import get
from time import sleep

from pprint import pprint

import sys, re, os

search_url = 'https://eol.org/search?utf8=%E2%9C%93&q={}'
data_url = 'https://eol.org{}/data'
eol_url = 'https://eol.org{}'
media_url = 'https://eol.org/pages/{}/media'

def get_gallery(url, index):
    url = url + '?page={}'.format(index)
    html = get(url).text
    processed = BeautifulSoup(html, 'html.parser')
    gallery   = processed.find(attrs={'id': 'gallery'})
    images    = [image.get('src') for image in gallery.find_all('img')]
    return images

def get_media_page(i):
    url = media_url.format(i)
    html = get(url).text
    processed = BeautifulSoup(html, 'html.parser')
    n_pages   = int(processed.find(attrs={'class' : 'last'}).find('a').get('href').split('=')[-1])
    galleries = []
    for x in range(n_pages):
        print(x)
        galleries.append(get_gallery(url, x))
    return galleries

def get_data_page(query):
    expanded = '+'.join(query.split()) + '+'
    url = search_url.format(expanded)

    html = get(url).text
    processed = BeautifulSoup(html, 'html.parser')
    top = processed.find(attrs={'class' : 'search-result'})
    link = top.find('a')
    nav = link.get('href')
    return nav


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

if __name__ == '__main__':
    images = get_media_page(30051)
    total = sum(len(x) for x in images)
    print(total)
    print(get_data_page('Encephalartos'))
