from petal.pipeline.module_utils.module import Module

class WikipediaModule(Module):
    '''
    This is the Module for downloading Wikipedia articles for a particular species.
    Its type signature takes a Species and returns a WikipediaArticle:Article, as shown below.
    In neo4j, these two nodes will be connected by the specified labels as well.
    After this is defined, all that matters is the process() function, which takes a Species node and returns a list of Transaction objects that will add WikipediaArticle nodes to the database.
    '''
    def __init__(self, in_label='Taxon', out_label='WikipediaArticle:Article', connect_labels=('MENTIONED_IN_ARTICLE', 'MENTIONS_SPECIES'), name='Wikipedia'):
        Module.__init__(self, in_label, out_label, connect_labels, name)
        self.SCRAPE_FIELDS = {'content', 'summary', 'links', 'references', 'images', 'title', 'url'}

    def process(self, previous):
        import wikipedia
        import datetime
        import requests
        import warnings
        with warnings.catch_warnings(record=True) as w: # Wikipedia module author has unpatched and easily fixed warnings :(
            # Lookup the species based on its name. Make sure that all Species objects have this attribute!!
            name = previous.data['name']
            try:
                results = wikipedia.search(name)
                for result in results: # Create a transaction for each result
                    result_properties = dict() # Properties for the neo4j node, populated below
                    # A lot of weird things can happen when crawling Wikipedia, so exception handling galore
                    try:
                        page = wikipedia.page(result, auto_suggest=True, redirect=True, preload=True) # Use wikiAPI to load the actual page
                        for field in self.SCRAPE_FIELDS: # Store only the desired properties (above) in the node properties
                            try:
                                result_properties[field] = getattr(page, field)
                            except KeyError as e:
                                self.log.log(e)
                        result_properties['abstract'] = result_properties['summary'] # Re-name
                        self.log.log(result_properties)
                        yield self.default_transaction(result_properties, uuid=page.title + '_WikipediaArticle', from_uuid=previous.uuid) # Only create default transaction objects
                    except KeyError as e:
                        self.log.log(e)
                    except wikipedia.exceptions.WikipediaException as e:
                        self.log.log(e)
                # In the future, use self.custom_transaction() and self.query_transaction() for more complicated Data Mining Modules
            except requests.exceptions.ConnectionError as e:
                self.log.log(e)

