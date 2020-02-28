from pprint import pprint
from subprocess import call
from time import time
import requests, zipfile, os
import os, pickle

def create_dir():
    # TODO: setup auto downloads from here by scraping most recent date?
    col_date = '2019-05-01' # Make sure this is a valid COL release
    if not os.path.isfile('data/.col_data/taxa.txt'):
        try:
            data = requests.get('http://www.catalogueoflife.org/DCA_Export/zip-fixed/{}-archive-complete.zip'.format(col_date))
            with open('col.zip', 'wb') as outfile:
                outfile.write(data.content)
            with zipfile.ZipFile('col.zip', 'r') as zip_handle:
                zip_handle.extractall('data/.col_data')
        except:
            if os.path.isfile('col.zip'):
                os.remove('col.zip')
            shutil.rmtree('data/.col_data')

def build_catalog():
    if os.path.isfile('catalog.pkl'):
        with open('catalog.pkl', 'rb') as infile:
            return pickle.load(infile)
    else:
        catalog = dict()

        create_dir() # Call the code above to download COL data if it isn't already present
        start = time()
        i = 0
        with open('data/.col_data/taxa.txt', 'r', encoding='utf-8') as infile:
            headers = None
            json    = dict()
            # Parse lines of the downloaded file, and add it as a default_transaction() (see yield statement)
            for line in infile:
                if i == 0:
                    headers = line.split('\t')
                    headers = ('id',) + tuple(headers[1:])
                else:
                    for k, v in zip(headers, line.split('\t')):
                        json[k] = v
                    try:
                        json.pop('isExtinct\n')
                    except KeyError:
                        pass
                    if json['taxonRank'] == 'species':
                        json['name'] = json['scientificName'].replace(json['scientificNameAuthorship'], '').strip()
                        catalog[json['name']] = json
                    json = dict()
                if i % 1000 == 0:
                    print('{}\r'.format(i), flush=True)
                i += 1
        with open('catalog.pkl', 'wb') as outfile:
            pickle.dump(catalog, outfile)
    return catalog

if __name__ == '__main__':
    catalog = build_catalog()
    print(len(catalog))
