
class WikipediaModule:
    def __init__(self, finder='MATCH (n:Species)', query='MATCH (n:Species) RETURN n', label='Article'):
        self.finder    = finder
        self.query     = query

    def process(self, tx, page):
        print(page)
