import wikipedia

from .module import Module

SCRAPE_FIELDS = {'content', 'summary', 'coordinates', 'links', 'references', 'images', 'title'}

class WikipediaModule(Module):
    def __init__(self, in_label='Species', out_label='WikipediaArticle:Article', connect_labels=('MENTIONED_IN_ARTICLE', 'MENTIONS_SPECIES'), name='Wikipedia'):
        Module.__init__(self, in_label, out_label, connect_labels, name)

    def process(self, node):
        name = node['name']
        try:
            results = wikipedia.search(name)
            properties = list()
            for result in results:
                print(result, flush=True)
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
                break
            return properties
        except Exception as e:
            print(e)
            return None

