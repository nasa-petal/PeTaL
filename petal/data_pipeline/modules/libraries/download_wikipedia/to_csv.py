import glob 
import xml.etree.ElementTree as etree
import xml

from pprint import pprint

def clean(title):
    return title.replace(' ', '_').replace('"', '').replace('?', '').replace('*', '(star)').replace("'", '').replace('/', '').replace(':', '_')

def to_tag(tag):
    return '{http://www.mediawiki.org/xml/export-0.10/}' + tag

def read_links(text):
    if text is None:
        return
    inside = False
    buff = ''
    for i in range(len(text) - 1):
        c1 = text[i]
        c2 = text[i + 1]
        if c1 == ']':
            if buff.strip() != '':
                yield buff
            buff = ''
            inside = False
        if inside and c1 != '[':
            buff += c1
        if c1 == '[' and c2 == '[':
            inside = True

def parse_page(page):
    properties = dict()

    properties['title'] = page.find(to_tag('title')).text
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
    properties = {k : v if v is not None else '' for k, v in properties.items()}
    page.clear()
    return properties

def to_json():
    link_queue = []
    page_tag = to_tag('page')
    with open('../../../data/enwiki-20200220-pages-articles-multistream.xml', "rb") as data:
        i = 0
        for event, elem in etree.iterparse(data, events=('end',)):
            if etree.iselement(elem) and elem.tag == page_tag:
                yield parse_page(elem)
                if i % 1000 == 0:
                    print(i, flush=True)
                i += 1
def to_csv():
    with open('pages.csv', 'w', encoding='utf-8') as pagefile:
        pagefile.write('title,text,redirect,links\n')
        with open('links.csv', 'w', encoding='utf-8') as linkfile:
            linkfile.write('from,to\n')
            for properties in to_json(): 
                try:
                    title = properties['title']
                    pagefile.write(properties['title'] + ',' + properties['redirect'] + '\n')
                    for link in properties['links']:
                        linkfile.write(title + ',' + link + '\n')
                    with open('pages/{}.txt'.format(clean(title)), 'w', encoding='utf-8') as outfile:
                        outfile.write(properties['text'])
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except Exception as e:
                    print(e, flush=True)

to_csv()

