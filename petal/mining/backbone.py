from bs4 import BeautifulSoup
from requests import get
from selenium import webdriver
from time import sleep

from pprint import pprint

from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotInteractableException

from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
neoDriver = GraphDatabase.driver(uri, auth=("neo4j", "life"))

import sys, re, os

from pprint import pprint

year = '2019'
url = 'https://www.catalogueoflife.org/annual-checklist/{}/browse/tree'.format(year)
cache_file = 'cached_catalogue_of_life.html'

sleep_time = 0.01

def add_species(tx, properties, pair):
    catalog_source, name = pair
    species_info = 'Name: {name},CatalogSource: {catalog_source}'
    taxa_info = ','.join('{key}:{{{key}}}'.format(key=k) for k in properties)
    prop_field = '{' + taxa_info + ', ' +  species_info + '}'
    query = 'CREATE (n:Species ' + prop_field + ')'
    # print(query, flush=True)
    # pprint(properties)
    tx.run(query, name=name, catalog_source=catalog_source, **properties)

def load_click(node):
    try:
        node.click()
        sleep(sleep_time)
    except ElementNotInteractableException:
        sleep(sleep_time)
        load_click(node)

def get_parent(node):
    try:
        return node.find_element_by_xpath('../..')
    except:
        return node

def expand(node):
    try:
        load_click(node)
    except AttributeError as e: # Can't click top level
        pass
    parent = get_parent(node)
    return parent.find_elements_by_class_name('dijitTreeExpandoClosed')

def parse_tag(element):
    tag = element.get_attribute('outerHTML')
    return tag.split('>')[1].split('<')[0]

def recursive_expand(node, depth=0, properties=None, session=None):
    if properties is None:
        properties = dict()
    parent = get_parent(node)
    rank, name = get_name_tuple(node, depth)
    rank = rank.strip()
    if rank == '':
        rank = 'Kingdom'
    properties[rank] = name
    children = expand(node)
    # Terminal case
    if len(children) == 0:
        with neoDriver.session() as session:
            links = parent.find_elements_by_tag_name('a')
            for i in range(len(links) // 2):
                pair = (name, parse_tag(links[i + 1]))
                session.read_transaction(add_species, properties, pair)
    # Normal case
    else:
        for child in children:
            found = recursive_expand(child, depth=depth + 1, properties=properties, session=session)
            if depth < 2:
                break
    # Collapse to save resources
    properties.pop(rank) 
    if depth > 0:
        node.click()

def get_name_tuple(node, depth=0, prefix='node-'):
    parent = get_parent(node)
    name   = parent.find_element_by_class_name('nodeLabel')
    rank   = parent.find_element_by_class_name('rank')
    return parse_tag(rank), parse_tag(name)

def main():
    driver = webdriver.Firefox()
    driver.get(url)
    recursive_expand(driver, depth=0)

if __name__ == '__main__':
    sys.exit(main())


