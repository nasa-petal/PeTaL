#python read xml files
import csv
import nltk
import glob, os
import requests
import xml.etree.ElementTree as ET
from os import listdir
from neo4j import GraphDatabase, basic_auth
#from py2neo import Graph, Node, Relationship
import codecs
from neomodel import (config, StructuredNode, StringProperty, IntegerProperty,
    UniqueIdProperty, RelationshipTo)

driver = GraphDatabase.driver("bolt://localhost:11002", auth=basic_auth("default", "testing"))

print(os.getcwd())
for file in glob.glob('C:/Users/dawillin/Documents/petal-master/petal2/petal/scrape-n-rank/highwire/*.xml'):
#if you have to be more selective inside your directory
#just add a conditional to skip here
    with open(file, "rb") as data:
        tree = ET.parse(data)
        root = tree.getroot()

        article_type = root.get('article-type')
        journal_title = tree.find('front').find('journal-meta').find('journal-title').text
        article_title = tree.find('front').find('article-meta').find('title-group').find('article-title').text
        publisher_name = tree.find('front').find('journal-meta').find('publisher').find('publisher-name').text
        pub_date = tree.find('front').find('article-meta').find('pub-date').find('year').text

try:
            issue_title = tree.find('front').find('article-meta').find('issue-title').text
except ex:
            issue_title=""

url = tree.find('front').find('article-meta').find('self-uri').text
abstract = tree.find('front').find('article-meta').find('abstract').find('p').text

possible_doi = tree.find('front').find('article-meta').findall('article-id')
doi =""
for x in possible_doi:
    if x.attrib['pub-id-type'] =="doi":
                doi = x.text

session = driver.session()
create_str ='CREATE (p:article {{articleTitle:\'{0}\', journalTitle:\'{1}\', articleType:\'{2}\', publisherName:\'{3}\', pubDate:\'{4}\', issueTitle:\'{5}\', url:\'{6}\', abstract:\'{7}\', doi:\'{8}\'}})'.format(article_title,journal_title,article_type,publisher_name,pub_date,issue_title,url,abstract,doi)
results = session.run(create_str)
session.close()
#import data to graph

#nodes =Node(name=article_title, articletype=article_type, )


