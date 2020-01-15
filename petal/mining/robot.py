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
        expandable = driver.find_elements_by_class_name('dijitTreeExpando')
        for _ in range(6):
            for exp in expandable[1:]:
                if 'Closed' in exp.get_attribute('class'):
                    exp.click()
            expandable = driver.find_elements_by_class_name('dijitTreeExpando')
        html = driver.page_source
        # Cache HTML
        with open(cache_file, 'w') as outfile:
            outfile.write(html)
        driver.quit()

if __name__ == '__main__':
    sys.exit(main())
