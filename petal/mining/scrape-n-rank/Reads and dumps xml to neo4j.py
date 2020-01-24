#python read xml files
import csv 
import glob, os
import requests 
import io
import xml.etree.ElementTree as ET
from os import listdir
from neo4j import GraphDatabase, basic_auth
import re 
import sys
from xml.sax.saxutils import escape
from neobolt.exceptions import CypherSyntaxError

driver = GraphDatabase.driver("bolt://139.88.179.199:7687", auth=basic_auth("neo4j", "testing"))
 
# xmltext = re.sub(u"[^\x20-\x7f]+",u"",xmltext)
print(os.getcwd())
errorList = []
xmlErrorFiles = []
indx = 1
for file in glob.glob('highwire/*.xml'):
    #if you have to be more selective inside your directory
    #just add a conditional to skip here
    try:
        with open(file, "rb") as data:
            tree = ET.parse(data)
            root = tree.getroot()
 
            article_type = root.get('article-type')
            journal_title = tree.find('front').find('journal-meta').find('journal-title').text
            article_title = tree.find('front').find('article-meta').find('title-group').find('article-title').text
            publisher_name = tree.find('front').find('journal-meta').find('publisher').find('publisher-name').text
            pub_date = tree.find('front').find('article-meta').find('pub-date').find('year').text
            url = tree.find('front').find('article-meta').find('self-uri').text
            try:
                abstract = tree.find('front').find('article-meta').find('abstract').find('p').text
            except:
                abstract = ""
            
            possible_doi = tree.find('front').find('article-meta').findall('article-id')
            doi = ""
            for x in possible_doi:
                if x.attrib['pub-id-type'] == "doi":
                    doi = x.text
                    
            create_str = 'CREATE (p:article{{articleTitle:\'{0}\', journalTitle:\'{1}\', articleType:\'{2}\', publisherName:\'{3}\', pubDate:\'{4}\', url:\'{5}\', doi:\'{6}\', abstract:\'{7}\'}})'.format(article_title,journal_title,article_type,publisher_name,pub_date,url,doi,abstract.replace('\'','\\\''))
            try:
                session = driver.session()
                indx+=1
                results = session.run(create_str)
                session.close()
            except Exception as e: 
                raise e
                print("Missing File {0}, Count {1}".format(file,indx), file=sys.stderr)
                errorList.append(create_str)
            else:
                print("Added File {0}, Count {1}".format(file,indx))
 
    except:
        print("Missing File {0}, Count {1}".format(file,indx), file=sys.stderr)
        xmlErrorFiles.append(file)

with open("output.txt", "w") as txt_file:
        for line in errorList:
            txt_file.write(" ".join(line) +"\n\n") # works with any number of elements in a line
 
with open("output_xml.txt", "w") as txt_file:
        for line in xmlErrorFiles:
            txt_file.write(" ".join(line) +"\n\n") # works with any number of elements in a line
