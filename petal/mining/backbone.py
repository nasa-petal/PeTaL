from bs4 import BeautifulSoup
from requests import get
from selenium import webdriver
from time import sleep

from pprint import pprint

from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotInteractableException

import sys, re, os

import pickle
import json

year = '2019'
url = 'https://www.catalogueoflife.org/annual-checklist/{}/browse/tree'.format(year)
cache_file = 'cached_catalogue_of_life.html'

sleep_time = 0.01

name_map = ['', 'phylum', 'class', 'order', 'family', 'genus', 'genus']

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
    except AttributeError as e:
        print(e)
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
    if len(children) == 0:
        links = parent.find_elements_by_tag_name('a')
        for i in range(len(links) // 2):
            pair = (name, parse_tag(links[i + 1]))
            collected.append(pair)
    # Normal case
    else:
        for child in children:
            found = recursive_expand(child, depth=depth+1)

            collected.extend([(name,) + f for f in found])
    # Collapse to save resources
    node.click()
    return collected

def get_name(node, depth=0, prefix='node-'):
    parent   = get_parent(node)
    element  = parent.find_element_by_class_name('nodeLabel')#prefix + name_map[depth])
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
        recursive_expand(driver, depth=0)

if __name__ == '__main__':
    sys.exit(main())
