from bs4 import BeautifulSoup
from requests import get
from selenium import webdriver
from time import sleep

from pprint import pprint

import sys, re, os

year = '2019'
url = 'https://www.catalogueoflife.org/annual-checklist/{}/browse/tree'.format(year)
cache_file = 'cached_catalogue_of_life.html'

sleep_time = 0.1

name_map = ['', 'phylum', 'class', 'order', 'family', 'genus', 'genus']

def get_parent(node):
    try:
        return node.find_element_by_xpath('../..')
    except:
        return node

def expand(node):
    try:
        node.click()
        sleep(sleep_time)
    except AttributeError:
        pass
    parent = get_parent(node)
    return parent.find_elements_by_class_name('dijitTreeExpandoClosed')

def parse_tag(element):
    tag = element.get_attribute('outerHTML')
    return tag.split('>')[1].split('<')[0]

def recursive_expand(node, depth=0):
    collected = []
    parent = get_parent(node)
    name = get_name(node, depth)
    children = expand(node)
    # Terminal case
    if depth >= 5:
        links = parent.find_elements_by_tag_name('a')
        for i in range(len(links) // 2):
            pair = (name, parse_tag(links[i + 1]))
            collected.append(pair)
    # Normal case
    else:
        for child in children:
            found = recursive_expand(child, depth=depth+1)

            collected.extend([(name,) + f for f in found])
    # Collapse kingdom and phylum after use to save resources
    if depth < 2 and depth > 0:
        node.click()
    return collected

def get_name(node, depth=0, prefix='node-'):
    parent   = get_parent(node)
    element  = parent.find_element_by_class_name(prefix + name_map[depth])
    return parse_tag(element)

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
                    #pprint(class_species_long_form)
                    print(class_species_long_form[-1])
                    1/0
                phylum.click()
            kingdom.click()

        return
        #html = driver.page_source
        # Cache HTML
        with open(cache_file, 'w') as outfile:
            outfile.write(html)
        driver.quit()

if __name__ == '__main__':
    sys.exit(main())
