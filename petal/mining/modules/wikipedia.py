import wikipedia

from .module import Module

SCRAPE_FIELDS = {'content', 'summary', 'coordinates', 'links', 'references', 'images', 'title'}

class WikipediaModule(Module):
    def __init__(self, in_label='Species', out_label='WikipediaArticle', connect_labels=('MENTIONED_IN_ARTICLE', 'MENTIONS_SPECIES')):
        Module.__init__(self, in_label, out_label, connect_labels)

    def process(self, node):
        name = node['Name'] if 'Name' in node else node['name']
        results = wikipedia.search(name)
        properties = list()
        for result in results:
            result_properties = dict()
            try:
                page = wikipedia.page(result, auto_suggest=True, redirect=True, preload=True)
                for field in SCRAPE_FIELDS:
                    try:
                        result_properties[field] = getattr(page, field)
                    except KeyError:
                        pass
            except wikipedia.exceptions.WikipediaException as e:
                print(e)
            properties.append(result_properties)
        return properties

