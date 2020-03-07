from pprint import pprint

from bs4 import BeautifulSoup
from requests import get
from time import sleep

from pprint import pprint

import sys, re, os

search_url = 'https://eol.org/search?utf8=%E2%9C%93&q={}'
media_url = 'https://eol.org/pages/{}/media'

def get_page_ids(query):
    expanded = '+'.join(query.split())
    url = search_url.format(expanded)
    html = get(url).text
    processed = BeautifulSoup(html, 'html.parser')
    top = processed.find(attrs={'class' : 'search-result'})
    if top is None:
        return
    links = top.find_all('a')
    for link in links:
        nav = link.get('href')
        yield int(nav.split('/')[-1])

def get_gallery(url, index):
    url = url + '?page={}'.format(index)
    html = get(url).text
    processed = BeautifulSoup(html, 'html.parser')
    gallery   = processed.find(attrs={'id': 'gallery'})
    for image in gallery.find_all('img'):
        yield image.get('src')

def get_media_page(i, display=False):
    url = media_url.format(i)
    try:
        html = get(url).text
        processed = BeautifulSoup(html, 'html.parser')
        n_pages   = int(processed.find(attrs={'class' : 'last'}).find('a').get('href').split('=')[-1])
        for x in range(n_pages):
            if display:
                print(x + 1, '/', n_pages, ' pages ', flush=True)
            yield get_gallery(url, x)
    except AttributeError: # Extend
        pass

def get_images(query, display=False):
    for page_id in get_page_ids(query):
        yield get_media_page(page_id, display=display)

if __name__ == '___main__':
    for page in get_images('Encephalartos'):
        print(sum([len(x) for x in page]))
