import glob 
import xml.etree.ElementTree as ET

from .module import Module

class HighwireModule(Module):
    def __init__(self, in_label=None, out_label='HighwireArticle', connect_label=None, name='Highwire'):
        Module.__init__(self, in_label, out_label, connect_label, name)

    def process(self):
        for file in glob.glob('highwire/*.xml'):
            try:
                with open(file, "rb") as data:
                    tree = ET.parse(data)
                    root = tree.getroot()
                    properties = dict()
                    properties['article_type']   = root.get('article-type')
                    properties['journal_title']  = tree.find('front').find('journal-meta').find('journal-title').text
                    properties['article_title']  = tree.find('front').find('article-meta').find('title-group').find('article-title').text
                    properties['publisher_name'] = tree.find('front').find('journal-meta').find('publisher').find('publisher-name').text
                    properties['pub_date']       = tree.find('front').find('article-meta').find('pub-date').find('year').text
                    properties['url']            = tree.find('front').find('article-meta').find('self-uri').text
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
                    yield properties
                    break
            except Exception as e:
                print(e)
