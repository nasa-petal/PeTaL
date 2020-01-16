from bs4 import BeautifulSoup
from requests import get
from selenium import webdriver
from time import sleep

from pprint import pprint

import sys, re, os

year = '2019'
url = 'https://www.catalogueoflife.org/annual-checklist/{}/browse/tree'.format(year)
cache_file = 'cached_catalogue_of_life.html'

sleep_time = 1

name_map = ['', 'phylum', 'class', 'order', 'family', 'genus']

def get_parent(node):
    return node.find_element_by_xpath('../..')

def expand(node):
    node.click()
    sleep(sleep_time)
    parent = get_parent(node)
    return parent.find_elements_by_class_name('dijitTreeExpandoClosed')

def recursive_expand(node, depth=0):
    name = get_name(node, depth)
    # Special cases for first two levels and for last level
    collected = []
    children = expand(node)
    for child in children:
        found = recursive_expand(child, depth=depth+1)
        collected.extend([(name,) + f for f in found])
    return collected

def get_name(node, depth=0, prefix='node-'):
    parent   = get_parent(node)
    element  = parent.find_element_by_class_name(prefix + name_map[depth])
    spanHTML = element.get_attribute('outerHTML')
    return spanHTML.split('>')[1].split('<')[0]

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

        kingdoms = driver.find_elements_by_class_name('dijitTreeExpandoClosed')
        for kingdom in kingdoms:
            k_name = get_name(kingdom, depth=0)
            phylums = expand(kingdom)
            for phylum in phylums:
                p_name = get_name(phylum, depth=1)
                classes = expand(phylum)
                for c in classes:
                    class_species_long_form = recursive_expand(c, depth=2) # Depth is a starting parameter
                    print(class_species_long_form)
                phylum.click()
            kingdom.click()

        return
        html = driver.page_source
        # Cache HTML
        with open(cache_file, 'w') as outfile:
            outfile.write(html)
        driver.quit()

if __name__ == '__main__':
    sys.exit(main())
