from requests import get
from bs4 import BeautifulSoup

from petal.pipeline.module_utils.module import Module

TOOLS_URL  = "http://airfoiltools.com"
SEARCH_URL = "http://airfoiltools.com/search/airfoils"

def scrape_airfoil_list():
    raw_html = get(SEARCH_URL).content
    html = BeautifulSoup(raw_html, 'html.parser')
    airfoilURLList = html.findAll("table", {"class": "listtable"})
    tableRows = airfoilURLList[0].findAll("tr")
    urls = []
    names = []
    for row in tableRows: # Search through all tables 
        airfoil_link = row.find(lambda tag: tag.name=="a" and tag.has_attr('href'))
        if (airfoil_link):
            urls.append(TOOLS_URL + airfoil_link['href'])
            names.append(airfoil_link.text.replace("\\", "_").replace("/","_"))
    return zip(urls, names)


class AirfoilList(Module):
    def __init__(self, in_label=None, out_label='AirfoilURL', connect_labels=None, name='AirfoilList'):
        Module.__init__(self, in_label, out_label, connect_labels, name)

    def process(self):
        for url, name in scrape_airfoil_list():
            yield self.default_transaction(dict(name=name, url=url), uuid=url)
