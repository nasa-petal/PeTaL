from neo4j import GraphDatabase, basic_auth
from neo import page

import json

from modules import WikipediaModule

class Driver():
    def __init__(self, page_size, rate_limit):
        self.neo_client = GraphDatabase.driver("bolt://139.88.179.199:7687", auth=basic_auth("neo4j", "testing"))
        self.page_size  = page_size
        self.rate_limit = rate_limit

    def paging(self, tx, module):
        for page_results in page(tx, module.finder, module.query, page_size=self.page_size, rate_limit=self.rate_limit):
            with self.neo_client.session() as session:
                session.write_transaction(module.process, page_results)

    def run(self, module):
        with self.neo_client.session() as session:
            session.read_transaction(self.paging, module)

if __name__ == '__main__':
    driver = Driver(10, 0.25)
    wiki_scraper = WikipediaModule()
    driver.run(wiki_scraper)
