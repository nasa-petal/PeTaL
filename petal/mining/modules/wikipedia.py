from neo import add_json_node

import wikipedia

SCRAPE_FIELDS = {'content', 'summary', 'coordinates', 'links', 'references', 'images', 'title'}

def get_wiki_page(page_name):
    properties = dict()
    try:
        page = wikipedia.page(page_name, auto_suggest=True, redirect=True, preload=True)
        for field in SCRAPE_FIELDS:
            try:
                properties[field] = getattr(page, field)
            except KeyError:
                pass
    except wikipedia.exceptions.WikipediaException as e:
        print(e)
    return properties

class WikipediaModule:
    def __init__(self, finder='MATCH (n:Species)', query='MATCH (n:Species) RETURN n', label='Article'):
        self.finder    = finder
        self.query     = query

    def process(self, tx, page):
        for record in page.records():
            node = record.get('n')
            name = node['Name'] if 'Name' in node else node['name']
            results = wikipedia.search(name)
            if len(results) > 0:
                a_node = add_json_node(tx, label='WikipediaArticle', properties=get_wiki_page(results[0]))
                a_node = list(a_node.records())[0].get('n')
                print('Added {}\'s page'.format(name), flush=True)
                result = tx.run('MATCH (n:Species) WHERE n.name = {name} MATCH (m:WikipediaArticle) WHERE m.title = {title} CREATE (m)-[:MENTIONS_SPECIES]->(n) RETURN m, n', name=name, title=a_node['title'])
                print('Added connection ', list(result.records()), flush=True)
        print('Finished page', flush=True)
