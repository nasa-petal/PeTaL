from bitflow.utils.module import Module

# Import Elasticsearch package 
from elasticsearch import Elasticsearch


# def create_index(client):
#     """Creates an index in Elasticsearch if one isn't already there."""
#     client.indices.create(
#         index="articles",
#         body={
#             "settings": {"number_of_shards": 1},
#             "mappings": {
#                 "properties": {
#                     "title": {"type": "text"},
#                     "abstract": {"type": "text"},
#                 }
#             },
#         },
#         ignore=400,
#     )


class ElasticSearchIndexer(Module):
    '''
    Creates an index in Elastic Search and adds articles as documents
    '''
    def __init__(self, in_label='Article', name='ElasticSearchIndexer'):
        Module.__init__(self, in_label, name, page_batches=True)

    def process(self, previous):
        '''
        :param previous: neo4j transaction representing an article
        '''

        # Connect to the elastic cluster
        #client = Elasticsearch('http://localhost:9200')
        client=Elasticsearch([{'host':'localhost','port':9200}])

        print("Creating an index if it one doesnt exist already...")
        # create_index(client)

        body = {
            "title":previous.data["title"],
            "abstract":previous.data["abstract"]
        }
        uuid = previous.data["uuid"]

        print("Indexing documents...")
        client.index(index="articles", body=body, id=uuid)
