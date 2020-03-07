import glob 
import xml.etree.ElementTree as etree
import xml

from pprint import pprint

from ..utils.module import Module

def to_tag(tag):
    return '{http://www.mediawiki.org/xml/export-0.10/}' + tag

def read_links(text):
    inside = False
    buff = ''
    for i in range(len(text) - 1):
        c1 = text[i]
        c2 = text[i + 1]
        if c1 == ']':
            yield buff
            buff = ''
            inside = False
        if inside and c1 != '[':
            buff += c1
        if c1 == '[' and c2 == '[':
            inside = True

class DownloadedWikipediaModule(Module):
    def __init__(self, in_label=None, out_label='WikipediaPage', connect_label=None, name='DownloadedWikipedia'):
        Module.__init__(self, in_label, out_label, connect_label, name)

    def parse_page(self, page):
        properties = dict()

        properties['title'] = page.find(to_tag('title')).text
        properties['uuid']  = properties['title']
        redirect = page.find(to_tag('redirect'))
        if redirect is None:
            revision = page.find(to_tag('revision'))
            properties['redirect'] = ''
            properties['text']  = revision.find(to_tag('text')).text
            properties['links'] = list(read_links(properties['text']))
        else:
            properties['redirect'] = redirect.attrib['title']
            properties['text'] = ''
            properties['links'] = [properties['redirect']]
        page.clear()
        return properties

    def process(self):
        with open('data/links.txt', 'w', encoding='utf-8') as linkfile:
            linkfile.write('')
        link_queue = []
        page_tag = to_tag('page')
        with open('data/enwiki-20200220-pages-articles-multistream.xml', "rb") as data:
            for event, elem in etree.iterparse(data, events=('end',)):
                if etree.iselement(elem) and elem.tag == page_tag:
                    properties = self.parse_page(elem)
                    with open('data/links.txt', 'a', encoding='utf-8') as linkfile:
                        linkfile.write((properties['title'] + ',' +  ','.join(properties['links']) + '\n'))
                    yield self.default_transaction(properties, uuid=properties['uuid'])
        print('Writing links!', flush=True)
        with open('data/links.txt', 'r', encoding='utf-8') as linkfile:
            for line in linkfile:
                page_title, *links = line.split(',')
                for link in links:
                    yield self.custom_transaction(uuid=link, from_uuid=page_title, connect_labels=('mentions', 'mentions'))
