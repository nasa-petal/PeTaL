import glob 
import xml.etree.ElementTree as ET

from pprint import pprint

from ..module_utils.module import Module

class HighwireModule(Module):
    def __init__(self, in_label=None, out_label='HighwireArticle:Article', connect_label=None, name='Highwire'):
        Module.__init__(self, in_label, out_label, connect_label, name)

    def process(self):
        for file in glob.glob('data/highwire/*.xml'):
            with open(file, "rb") as data:
                try:
                    tree = ET.parse(data)
                    root = tree.getroot()
                    properties = dict()
                    front = tree.find('front')
                    article_meta = front.find('article-meta')
                    journal_meta = front.find('journal-meta')
                    properties['type']      = root.get('article-type')
                    properties['url']       = article_meta.find('self-uri').text
                    properties['date']      = article_meta.find('pub-date').find('year').text
                    properties['title']     = article_meta.find('title-group').find('article-title').text
                    properties['journal']   = journal_meta.find('journal-title').text
                    properties['publisher'] = journal_meta.find('publisher').find('publisher-name').text
                    try:
                        properties['abstract'] = tree.find('front').find('article-meta').find('abstract').find('p').text
                    except:
                        properties['abstract'] = ""
                    possible_doi = tree.find('front').find('article-meta').findall('article-id')
                    doi = ""
                    for x in possible_doi:
                        if x.attrib['pub-id-type'] == "doi":
                            doi = x.text
                    properties['doi'] = doi
                    skip = False
                    for v in properties.values():
                        if v is None:
                            skip = True
                    if not skip:
                        yield self.default_transaction(properties)
                except xml.etree.ElementTree.ParseError:
                    pass
