import wikipedia as wiki
import re

def clean(html):
    return re.findall(r"[\w']+",html)[9].lower()

def get_phylum(cls):
    html = wiki.page(cls).html()
    ix = html.find('Phylum')
    if ix == -1:
        return ""
    return clean(html[ix:ix+100])
def get_kingdom(cls):
    html = wiki.page(cls).html()
    ix = html.find('kingdom')
    if ix == -1:
        return ""
    return clean(html[ix:ix+100])