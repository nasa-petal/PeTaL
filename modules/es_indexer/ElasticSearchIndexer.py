from bitflow.utils.module import Module
from elasticsearch import Elasticsearch

class ElasticSearchIndexer(Module):
    '''
    A simple placeholder indexer for Elastic search
    '''
    def __init__(self, in_label='Article', name='ElasticSearchIndexer'):
        Module.__init__(self, in_label, name, page_batches=True)

    def process(self, previous):
        '''
        :param previous: neo4j transaction representing an article
        '''

        # Connect to the elastic cluster
        client=Elasticsearch([{'host':'localhost','port':9200}])


        body = {
            "title":previous.data["title"],
            "abstract":previous.data["abstract"]
        }
        uuid = previous.data["uuid"]

        print("Indexing documents...")
        client.index(index="articles", body=body, id=uuid)
