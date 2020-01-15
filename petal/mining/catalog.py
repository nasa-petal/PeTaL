from bs4 import BeautifulSoup
from requests import get
from selenium import webdriver

import sys, re

year = '2019'

def main():
    url = 'https://www.catalogueoflife.org/annual-checklist/{}/browse/tree'.format(year)
    driver = webdriver.Firefox()
    driver.get(url)
    html = driver.page_source
    #catalog = get('https://www.catalogueoflife.org/annual-checklist/{}/browse/tree'.format(year))
    processed = BeautifulSoup(html, 'html.parser')
    body = processed.body
    test = body.find(id='right-col').find(id='content').find(id='tree')
    print(test)
    #print(dir(test))
    #print(catalog.text)
    #print(test.catalog)
    #print(processed.prettify())
    #print(dir(processed))
    #print(processed.findAll(id='tree'))
    #print(processed.findAll(class='dijitTreeNode'))
    #print(processed.find_all('div', 'dijitTreeNode'))
    #print(processed.find_all(id='ACI_dojo_TxTreeNode_0'))

if __name__ == '__main__':
    sys.exit(main())
