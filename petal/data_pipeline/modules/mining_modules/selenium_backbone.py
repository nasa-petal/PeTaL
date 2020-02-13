from bs4 import BeautifulSoup
from requests import get
from selenium import webdriver
from time import sleep

from pprint import pprint

from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotInteractableException
from selenium.common.exceptions import ElementClickInterceptedException

from neo4j import GraphDatabase, basic_auth

# uri = "bolt://localhost:7687"
# neoDriver = GraphDatabase.driver(uri, auth=("neo4j", "life"))

neoDriver = GraphDatabase.driver("bolt://139.88.179.199:7687", auth=basic_auth("neo4j", "testing"))


from time import time

import sys, re, os

from pprint import pprint

from multiprocessing import Value, Process

year = '2019'
url = 'https://www.catalogueoflife.org/annual-checklist/{}/browse/tree'.format(year)
cache_file = 'cached_catalogue_of_life.html'

sleep_time = 0.01
start_time = time()

from neo import add_json_node

def add_species(tx, properties, pair):
    ext_properties = {k : v for k, v in properties.items()}
    ext_properties['Name'] = pair[1]
    ext_properties['CatalogSource'] = pair[0]
    add_json_node(tx, label='Species', properties=ext_properties)
    # catalog_source, name = pair
    # species_info = 'Name: {name},CatalogSource: {catalog_source}'
    # taxa_info = ','.join('{key}:{{{key}}}'.format(key=k) for k in properties)
    # prop_field = '{' + taxa_info + ', ' +  species_info + '}'
    # query = 'CREATE (n:Species ' + prop_field + ')'
    # print(query)
    # tx.run(query, name=name, catalog_source=catalog_source, **properties)

def load_click(node):
    try:
        node.click()
        sleep(sleep_time)
    except ElementClickInterceptedException:
        sleep(sleep_time * 10)
        node.click()
        # load_click(node)
    except ElementNotInteractableException:
        sleep(sleep_time * 10)
        node.click()
        # load_click(node)

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
    try:
        tag = element.get_attribute('outerHTML')
        return tag.split('>')[1].split('<')[0]
    except AttributeError as e:
        print(e)
        pass

def restart_to(node=None, properties=None):
    if node is None:
        driver = webdriver.Firefox()
        node = driver
        node.get(url)
    else:
        driver = None
    children = expand(node)
    if len(properties) == 0:
        node.click()
        return driver, node
    for child in children:
        name_tup = get_name_tuple(child)
        if name_tup in properties.items():
            properties.pop(name_tup[0])
            return driver, restart_to(node=child, properties=properties)[1]
    # print('Properties not found', properties)
    return driver, node

def get_kps(node, depth=0):
    if depth == 0:
        children = expand(node)
        return [item for phylum in [get_kps(child, depth=depth+1) for child in children] for item in phylum]
    else:
        rank, name = get_name_tuple(node)
        if depth == 2:
            return name
        if depth == 1:
            children = expand(node)
            return [(name, get_kps(child, depth=depth+1)) for child in children]

def recursive_expand(node, depth=0, properties=None, total=None):
    if properties is None:
        properties = dict()
    if depth != 0:
        parent = get_parent(node)
        rank, name = get_name_tuple(node)
        properties[rank] = name
    children = expand(node)
    # Terminal case
    if len(children) == 0:
        with total.get_lock():
            with neoDriver.session() as session:
                links = parent.find_elements_by_tag_name('a')
                for i in range(len(links) // 2):
                    j = i * 2
                    pair = (parse_tag(links[j]), parse_tag(links[j + 1]))
                    session.read_transaction(add_species, properties, pair)
                    total.value += 1
    # Normal case
    else:
        for child in children:
            total = recursive_expand(child, depth=depth + 1, properties=properties, total=total)
    # Collapse to save resources
    if depth > 0:
        properties.pop(rank) 
        node.click()
    return total

def get_name_tuple(node, prefix='node-'):
    try:
        parent = get_parent(node)
        name   = parent.find_element_by_class_name('nodeLabel')
        rank   = parent.find_element_by_class_name('rank')
        rank = parse_tag(rank).strip()
        name = parse_tag(name).strip()
        if rank == '':
            rank = 'Kingdom'
        return rank, name
    except NoSuchElementException:
        link = parent.find_element_by_tag_name('a')
        return 'SuperSpecies', parse_tag(link)


def run_kp(kp, total=None):
    kingdom, phylum = kp
    properties = {'Kingdom' : kingdom, 'Phylum' : phylum}
    driver, init = restart_to(properties=properties)
    total = recursive_expand(init, depth=2, total=total)
    driver.close()

def main():
    driver = webdriver.Firefox()
    driver.get(url)
    kps = get_kps(driver)
    driver.close()

    total = Value('i', 0)
    total.value = 0

    unstarted = []
    for kp in kps:
        unstarted.append(Process(target=run_kp, args=(kp,), kwargs={'total' : total}))

    max_running = 5
    running = []
    finished = []
    done = False
    while not done:
        sleep(.1)
        with total.get_lock():
            if total.value > 0:
                duration = time() - start_time
                rate = round(total.value/duration, 2)
                print('    ETA: {} hours'.format(1800000 / rate / 3600))
                print('    Processed {} species at {} species per second for {} total seconds'.format(total, rate, round(duration, 2)), flush=True)
        for item in running:
            if not item.is_alive():
                finished.append(item)
        running = [item for item in running if item.is_alive()]
        if len(running) < max_running:
            try:
                next_to_run = unstarted.pop()
                next_to_run.start()
                running.append(next_to_run)
            except IndexError:
                pass
        done = len(running) == 0 and len(unstarted) == 0

if __name__ == '__main__':
    sys.exit(main())


