from bs4      import BeautifulSoup 
from requests import get
from pprint import pprint

from .module import Module

JEB_LIMIT = 2

class JEBModule(Module):
    def __init__(self, in_label='Species', out_label='JEBArticle:Article', connect_labels=('MENTIONED_IN_ARTICLE', 'MENTIONS_SPECIES')):
        Module.__init__(self, in_label, out_label, connect_labels)

    def process(self, node):
        name   = node['name']
        url    = 'https://jeb.biologists.org/search/' + name.replace(' ', '%252B')
        print(url)
        print(name)
        result = get(url)
        soup   = BeautifulSoup(result.content)
        articles = []
        article_links = ['https://jeb.biologists.org' + x.get('href') for x in soup.find_all('a', attrs={'class': 'highwire-cite-linked-title'})]
        for i, article_link in enumerate(article_links):
            if i == JEB_LIMIT:
                break
            article_page = BeautifulSoup(get(article_link).content)
            title        = article_page.find(attrs={'class' : 'highwire-cite-title'}).get_text()
            abstract     = article_page.find(attrs={'class' : 'fulltext-view'}).get_text()
            authors      = article_page.find(attrs={'class' : 'highwire-cite-authors'}).get_text()
            print(title, flush=True)
            articles.append(dict(title=title, abstract=abstract, authors=authors, url=article_link))
        return articles
