from bs4 import BeautifulSoup
from requests import get
from selenium import webdriver

from pprint import pprint

import sys, re, os

year = '2019'
url = 'https://www.catalogueoflife.org/annual-checklist/{}/browse/tree'.format(year)
cache_file = 'cached_catalogue_of_life.html'

def main():
    # Load raw HTML of COL (ideally a one-off)
    # Caching = 50x speedup
    if os.path.isfile(cache_file):
        # Use cached HTML
        with open(cache_file, 'r') as infile:
            html = infile.read()
    else:
        driver = webdriver.Firefox()
        driver.get(url)
        html = driver.page_source
        # Cache HTML
        with open(cache_file, 'w') as outfile:
            outfile.write(html)
        driver.quit()
    # Begin parsing HTML of COL to get Species
    processed = BeautifulSoup(html, 'html.parser')
    body      = processed.body

    phylums = body.find_all(attrs={'class' : 'nodeLabel node-phylum'})
    print(phylums)
    return
    kingdoms = body.find_all(attrs={'class' : 'nodeLabel node-'})
    for kingdom in kingdoms:
        parent = kingdom.parent.parent.parent
        phylums = parent.find_all(attrs={'class' : 'nodeLabel node-phylum'})
        print(kingdom.text)
        for phylum in phylums:
            print('    ' + phylum.text)

if __name__ == '__main__':
    sys.exit(main())
