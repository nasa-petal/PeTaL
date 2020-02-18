from neo4j import GraphDatabase, basic_auth
import urllib.request
import urllib3
import requests
import os

neo_client = GraphDatabase.driver("bolt://139.88.179.199:7687", auth=basic_auth("neo4j", "testing"), encrypted=False)
with neo_client.session() as session:
    images = session.run('match (i:Image) return i')
for record in images.records():
    node = record['i']
    url  = node['url']
    uuid = node['uuid']
    ext = url.split('.')[-1]
    if ext not in {'svg'}:
        i = 0
        filename = 'data/images/{uuid}_{i}.'.format(uuid=str(uuid), i=str(i)) + ext
        while os.path.isfile(filename):
            i += 1
            filename = 'data/images/{uuid}_{i}.'.format(uuid=str(uuid), i=str(i)) + ext
        print(filename)
        try:
            urllib.request.urlretrieve(url, filename)
        except urllib3.exceptions.NewConnectionError:
            pass
        except TimeoutError:
            pass
        except urllib.error.HTTPError:
            pass
        except requests.exceptions.ConnectionError:
            pass
