"""
This program scrapes the information from the HighWire Current Repository and outputs a list of
full abstracts along with the journal that they are from.

Author: @Lauren E Friend  lauren.e.friend@nasa.gov
Editor: @Drayton Munster drayton.w.munster@nasa.gov
Note: this program only runs on Python 3.7 and above

**Note: HighWire Current access is granted on a person-by person basis. Please read the copyright information
on their website and apply for access via  highwirecurrent@highwire.org
"""


# Input the things
import nltk
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from nltk.stem.snowball import SnowballStemmer
import xml.etree.ElementTree as ET
import tarfile
from pathlib import Path
from dataclasses import dataclass
import pickle
from typing import List
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer(language="english", ignore_stopwords=True)
tokenizer = RegexpTokenizer(r'\w+')


# crate dataclass
@dataclass(frozen=True)
class Article:
    article_type: str
    journal_abrev: str
    journal_title: str
    article_title: str
    article_link: str
    doi: str
    abstract: str
    word_bag: List[str]
    species: List[str]

    @classmethod
    def from_highwire_xml(cls, xml_file):

        a_type = None
        abrev = None
        j_title = None
        a_title = None
        link = None
        art_doi = None
        a_abstract = None
        w_bag = None
        specs = None

        for (event, elem) in ET.iterparse(xml_file, events=['end']):  # iterparse goes through xml

            if elem.tag == 'article':  # obtains the information from the article tag
                a_type = elem.get('article-type')
            if elem.tag == 'journal-id':  # gets journal id_info (unique to HighWire Current)
                abrev = stringify(elem)
            if elem.tag == 'journal-title':  # gets the title of the journal
                j_title = stringify(elem)
            if elem.tag == 'article-title':  # gets the info from the article title xml tag as string
                a_title = stringify(elem)
            if elem.tag == 'self-uri':  # gets the link to the full article
                link = stringify(elem)
            if elem.tag == 'article-id':
                pub_type = elem.get('pub-id-type')
                if pub_type == "doi":
                    art_doi = stringify(elem)
            if elem.tag == 'abstract':  # gets the text in the abstract tag to keep for the user
                a_abstract = stringify(elem)
                w_bag = clean_article_word_bag(a_abstract)  # saves a copy of the abstract that will later be cleaned
                specs = animas(a_abstract, a_title)

        if a_type is None:  # making sure the article is not blank
            raise ValueError(f"Could not parse article type: Title: {a_title}")

        inst = cls(a_type, abrev, j_title, a_title, link, art_doi, a_abstract, w_bag, specs)  # turn the xml data to dataclass
        return inst


def stringify(art_elem):
    """
    This function takes in an Article, changes the xml tags to html tags to display to the user,
    and turns the xml information into a string.
    """
    substitutions = {'italic': 'i'}  # xml to html
    new_string = element_to_string(art_elem, trim_outer=True, tag_subs=substitutions).strip()  # turn to string
    return new_string


def element_to_string(elem: ET.Element, trim_outer: bool = False, tag_subs: dict = {}):
    """
    This function takes in an xml element in element tree format and tag substitutions as a
    dictionary. It then changes the xml to html tags and turns the xml element into text.
    """
    if trim_outer:
        full_text = []
    else:
        tag = elem.tag
        tag = tag_subs.get(tag, tag)
        full_text = [f"<{tag}>"]
    full_text.append(f'{elem.text or ""}')

    for child in elem:
        full_text.extend(element_to_string(child, trim_outer=False, tag_subs=tag_subs))

    if not trim_outer:
        full_text.append(f"</{tag}>")
    full_text.append(f'{elem.tail or ""}')

    return "".join(full_text)


def get_journals(j_fp):
    """
    This function reads in the journal abreviations from a separate text file separated by new lines and
    creates a list for later use.
    """
    des_journals = []
    with open(j_fp, "r") as f:  # open the text file in read mode
        for line in f:
            line = line.strip('\n')  # remove the \n character
            des_journals.append(line)  # add the journal abreviation to the list
    return frozenset(des_journals)


def get_tar_files(fp):
    """
    Takes in a folder path and outputs a list of all the Windows Path names of the .tar.gz files.
    """
    return Path(fp).glob("*.tar.gz")  # group all of the tar files together


def choose_files(files, des_jrnls):
    """
    Takes in the journal abbreviations and the list of windows path files and outputs a list of the windows
    file paths for the desired journals.
    """
    wanted_tars = []
    for file in files:
        abrev = file.name.split("_", maxsplit=1)[0]  # get just the abbreviation
        if abrev in des_jrnls:
            wanted_tars.append(file)  # only append the paths that are from approved journals
    return wanted_tars  # list of published issues from specific group of journals


# make sure the tar fies are not empty
def check_contents(des_files):
    """
    Takes in the list of desired journal windows paths, checks if they are empty, and outputs the
    non-empty tar files as a list.
    """
    tfile_list = []
    for file in des_files:
        tar_file = tarfile.open(file, "r:gz")
        xmls = tar_file.getnames()  # get the names of all the files the tar file contains
        if len(xmls) == 0:
            print(f"the tar file {file} has no contents.")  # error handle if the tar file is empty
            continue
        tfile_list.append(file)
    return tfile_list  # list of non-empty tar files


def get_xml(full_tars):
    """
    Takes in the list of full tar files WP, iterates through, opens, unzips, and outputs a list of ElementTree Elements.
    """
    xml_files = []
    for file in full_tars:
        tar = tarfile.open(file, "r:gz")  # open the tar file in zipped tar mode
        for member in tar:
            if not member.name.endswith(".xml"):  # if the contents are not an xml file skip them
                continue
            f = tar.extractfile(member)  # extract each xml file as an IOBufferedReader object
            if f is not None:
                xml_files.append(f)  # if the xml file is not empty, append it to a list to process
    return xml_files  # list of xml file instances as IO BufferedReader objects


def make_articles(xml_file_list):
    """
    Takes in a list of xml files, creates an Article instance for each, and returns a list of Articles.
    """
    article_list = []
    for item in xml_file_list:
        try:
            article = Article.from_highwire_xml(item)  # create an Article instance from the xml  
            if article.article_type == "research-article":  # only process research articles (not letters, etc.)
                if None not in (article.abstract, article.word_bag, article.article_title):
                    article_list.append(article)
        except ET.ParseError:  # many of the HighWire xml files are malformed. This error is okay in small quantities
            print(f"error parsing {item}")
        except FileNotFoundError:
            print(f"the file {item} does not exist")
    return article_list  # a list of Article instances from the HighWire repository


# saves the Article objects to a file
def pickle_checkpoint(articles, file_location, object_name):
    with open(f'desired_articles/scored_articles.pickle', "wb") as f:  # make a pickle to the right folder
        pickle.dump(articles, f)


def clean_article_word_bag(abstract):
    """
    this function takes a list of Articles, cleans the word_bag section, and returns
    a list of updated Article instances.
    Parameters
    ----------
    abstract: str

    Returns
    -------
    clean_wb_a: List[str]

    """
    wb = re.sub('<.*?>', "", abstract)  # remove ALL tags
    words = tokenizer.tokenize(wb)  # split the string into a list of words
    important_words = [w for w in words if w not in stop_words]
    new_words = [word.lower() for word in important_words]  # lower the words and remove punctuation
    q = [stemmer.stem(l_word) for l_word in new_words]  # stem & remove stopwords
    return q

def stringy_abstract(abstract):
    wub = re.sub('<.*?>', "", abstract)# remove ALL tags
    return wub

def animas(a_abstract, a_title):
    #finds species in abstracts
    species = [line.rstrip() for line in open("clean.txt", encoding = 'utf-8')]
    wubs = stringy_abstract(a_abstract)
    animal_articles = []
    for spec in species:
        my_regex = r"\b(?=\w)" + re.escape(spec) + r"\b(?!\w)"
        if re.search(my_regex, wubs) or re.search(my_regex, a_title):
            animal_articles.append(spec)
    if animal_articles:
        return animal_articles
    else:
        return None

def get_highwire_publishing_info(hwfp):
    """
    This is an optional function that takes the full directory of downloaded HighWire information
    and returns the journal title and the HighWire abbreviations for selection.
    Parameters
    ----------
    hwfp: str
        file path for the highwire repository information

    Returns
    -------

    """
    hw_jrnls = []
    for folder in get_tar_files(hwfp):
        abrev = folder.name.split("_", maxsplit=1)[0]
        if abrev not in hw_jrnls:
            hw_jrnls.append(abrev)
            try:
                with tarfile.open(folder, "r:gz") as tar_file:
                    for member in tar_file:
                        file = tar_file.extractfile(member)
                        for (event, elem) in ET.iterparse(file, events=['end']):
                            if elem.tag == 'journal-title':
                                title = stringify(elem)
                            if elem.tag == 'journal-id':
                                jabrev = stringify(elem)
                        break
            except ET.ParseError:
                print(f"error parsing {folder}")
            print(f"({jabrev},{title})")


def animal_research_test(argument):
    if argument == 'lfbbwt19':
        with open("data//0bbwt.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                print(line)


def get_articles_from_highwire_repository(journal_fp, hw_file_path, article_output_path):
    """
    This is the main function. It takes in a the file path containing the journal abbreviations, the
    HighWire Current repository file path, and the output file path for the pickled objects. It
    returns a list of Article instances processed from HighWire, and outputs a pickle.
    """
    j_abbreviations = get_journals(journal_fp)  # get the journal abbreviations
    print("proscessing HighWire documents now. Note: Some xml files have been created incorrectly and parsing errors are okay.")
    tar_files = check_contents(choose_files(get_tar_files(hw_file_path), j_abbreviations))  # list of good tar files
    xmls = get_xml(tar_files)  # list of xml files to process
    list_of_articles = make_articles(xmls)  # make Articles from the xml files
    pickle_checkpoint(list_of_articles, article_output_path, object_name="scored_articles")  # pickle out Articles
    return list_of_articles  # return a list of Article instances


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tars_path", default="./HighWire",
        help="The file path for the compressed HighWire archives"
    )
    parser.add_argument(
        "--article_output_path", default="./desired_articles",
        help="the file path for the intermediate proscessing results"
    )
    parser.add_argument(
        "--j_abrev_fp", default="journal_inputs.txt",
        help="File path for the journal abbreviations text file"
    )
    parser.add_argument("--bbwt", default="hello, world", help="this is optional, skip it")

    args = parser.parse_args()

    # main function
    get_articles_from_highwire_repository(args.j_abrev_fp, args.tars_path, args.article_output_path)

    # IF THIS IS YOUR FIRST TIME RUNNING BIRD, FIND JOURNALS VIA get_highwire_publishing_info
    # get_highwire_publishing_info(args.tars_path)

    # testcase: check your directory
    animal_research_test(args.bbwt)

