# -*- coding: utf-8 -*-
'''
Created on Wed Jul 11 07:52:04 2018

@author: bwhiteak and cbaumler
'''

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from spacy.matcher import Matcher
import pyLDAvis
import pyLDAvis.sklearn
import string, json, sys, pickle

ignore = pickle.load( open( "petal/data/cluster/ignore/ignore.p", "rb" ) )

def retrieve_topic_docs(topics, transformed_mat, n_take):
    '''
        Take a list of topics from topic_splitter and return all topic
        associated docs by using a sorted transformed_mat.
        e.g.
        given topics[3,7,1] return all docs containing 3,7,1 in their top
        three (non-supertopic)

        return a list of indexes
    '''
    ###Try catch here for index out of bounds... take==9 would get 10...
    ### take+1 > n_components
    ordered_idxs = (np.argsort(transformed_mat, axis=1))[:, 1:n_take + 1]
    return np.where(np.any(np.isin(ordered_idxs, topics), axis=1) == True)


# DATA FRAME BASED
def revised_subsetter(topic_sets, df):
    z = list(set(topic_sets[0]))
    return df.iloc[z], z


def make_doc_dict(doc_list, assoc_topics):
    '''
        Create a dictionary of the doc title with topic associations. This will be used for#        display to user.

        Return a dict key: docname value: list of top topics for doc
    '''
    doc_dict = {}  # Empty dict
    for i in range(len(doc_list)):  # Populate dict using loop
        key_str = "doc_%d" % i
        doc_dict[key_str] = assoc_topics[i]
    return doc_dict


def make_topic_dict(model_list, feature_names, n_top_words):
    '''
        This creates a dictionary containing:
        {'topic number':..., 'string of terms':...}

        Returns the dict
    '''
    msg_list = []
    all_list = [i.components_ for i in model_list]
    # all_mat = np.vstack((all_list[0], all_list[1]))
    all_mat = np.vstack((all_list[0]))
    for _, topic in enumerate(all_mat):
        message = " ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]])
        msg_list.append(message)
    return {i: x for i, x in enumerate(msg_list)}


def make_doc_list(df):
    '''
        This pulls the text from the selected indexes.  The text is used to
        form new tf and tfidf matrices and then new models are fitted on
        the set.

        Returns a list of text.
    '''
    return df['all'].tolist()


def make_subset_dict(df):
    x = df.index.values
    y = df['Title'].tolist()
    return dict(zip(x, y))


def get_lemma(token):
    if token.text == "species":
        # this is somewhat brittle, but the lemmatizer always maps 'species' to 'specie' which will likely always be wrong in biological texts
        return "species"
    return token.lemma_


def parse_tokens(term, term_token, tokens):
    '''
    Resursive function called by get_split_tokens to get singleton tokens as well
    as copmpounds and verb + direct object pairs

    Parameters
    ----------
    term : `str`
        The current term. May be made up of more than one token
    term_token : `spacy.tokens.token.Token`
        The current spacy token
    tokens :  `dict`
        The lists of n, v, vd, and all

    Returns
    ----------
    n, v, vd, all : `list` [`str`]
        Lists of the lemmatized nouns, verbs, verb + direct object pairs, and all
    '''
    term_token_lemma = get_lemma(term_token)
    term_token_head_lemma = get_lemma(term_token.head)

    if (term_token.dep_ == "compound" or (term_token.pos_ == "ADJ" and term_token.dep_ != "ROOT")) and term_token_lemma not in ignore['o']:
        if term_token.dep_ == "conj":
            # e.g. red and green birds should yeild red birds and green birds
            modified_term = term_token_lemma + " " + get_lemma(term_token.head.head)
            tokens = parse_tokens(modified_term, term_token.head.head, tokens)
        else:
            modified_term = term_token_lemma + " " + term_token_head_lemma
            tokens = parse_tokens(modified_term, term_token.head, tokens)
    elif term_token.dep_ == "dobj" and term_token.head.pos_ =="VERB" and term_token_lemma not in ignore['o']:
        modified_term = term_token_head_lemma + " " + term
        if "aux" not in term_token.head.dep_ and term_token_head_lemma not in ignore['v'] and modified_term not in ignore['vd']:
            tokens['vd'].append(modified_term)
            tokens['all'].append(modified_term)
            try:
                tokens['v'].remove(term)
                tokens['all'].remove(term)
            except:
                pass
    else:
        if (term_token.pos_=="NOUN" or term_token.pos_=="PROPN") and term_token_lemma not in ignore['n']:
            tokens['n'].append(term)
            tokens['all'].append(term)
        elif term_token.pos_=="VERB" and "aux" not in term_token.dep_ and term_token_lemma not in ignore['v']:
            tokens['v'].append(term)
            tokens['all'].append(term)

    return tokens


def get_split_tokens(doc):
    '''
    Makes list of lemmatized nouns (e.g. cell), verb (e.g. increase),
    verb + direct object pairs (e.g. increase temperature) and a list with all three.

    Parameters
    ----------
    text : `str`
        The text to be split into tokens

    Returns
    ----------
    n, v, vd, all : `list` [`str`]
        Lists of the lemmatized nouns, verbs, verb + direct object pairs, and all

    '''

    tokens = {'n':[], 'v':[], 'vd':[], 'all':[]}

    for token in doc:
        tokens = parse_tokens(get_lemma(token), token, tokens)

    return tokens['n'], tokens['v'],  tokens['vd'], tokens['all']

def match_n_merge(matcher, doc):
    '''
    Matches and merges spans in the document. The matched spans may overlap.

    Parameters
    ----------
    matcher : `spacy.matcher.matcher.Matcher`
        The matcher with some rule
    doc : `spacy.tokens.doc.Doc`
        The document to be used

    Returns
    ----------
    doc : `spacy.tokens.doc.Doc`
        The document with matched spans merged
    '''

    matches = matcher(doc)
    match_list = []

    if matches:
        start_span = matches[0][1]
        end_span = matches[0][2]

        for _, start, end in matches:
            if end_span > start:
                end_span = end
            else:
                match_list.append(doc[start_span:end_span])
                start_span = start
                end_span = end

        match_list.append(doc[start_span:end_span]) # merge the last span

        with doc.retokenize() as retokenizer:
            for span in match_list:
                retokenizer.merge(span)

    return doc

def create_df0(text):   # todo: currently is dependent on \r\n line endings
    '''
        Format of expected text string is:
        ------------------------------------------
        title\r\n
        abstract
        *
        *
        *
        text\r\n\r\n\r\n
        ------------------------------------------
        for each abstract in the file.

        This function recieves a single long string of abstracts. The split of the text is
        done on "\r\n\r\n\r\n". The string is parsed and the split abstracts are placed
        into a dataframe. Duplicates are dropped. From the dataframe we can extract
        a list and append it to an existing textfile abstracts.

        **Note: drop_duplicates() is used since apscheduler may have process missfire
        in which case we will get a duplicate abstract when it re-tries.  So we
        correct this in the df.

        Returns a list of abstracts
    '''
    if sys.platform == 'win32':
        line_sep = '\r\n'
    else:
        line_sep = '\n'
    text_list = [y.strip() for y in text.split(sep=(line_sep * 3))]  # Split abstracts str
    title_list = [x.split(line_sep, 1) for x in text_list]  # Break off titles
    if title_list[-1] == ['']:
        del title_list[-1]

    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("Cannot find spacy model. Try 'python -m spacy download en'")
    matcher = Matcher(nlp.vocab)

    matcher.add("hyphen", None, [{}, {"TEXT": "-"}, {}])

    for i in range(len(title_list)):
        if len(title_list[i]) > 1:
            doc = nlp(title_list[i][1])
            doc = match_n_merge(matcher, doc)
            n, v, vd, al = get_split_tokens(doc)
            title_list[i] = [title_list[i][0], n, v, vd, al]
    df_abstracts = pd.DataFrame(title_list, columns=['Title', 'n', 'v', 'vd', 'all'])  # Make df
    return df_abstracts


def cluster(df0, n_components):
    '''
    Creates the actual clusters. Called by create_cluster or recluster.

    Parameters
    ----------
    df0 : `DataFrame`
        generated by create_cluster. Can be either in dataframe format or JSON serialized
    n_components : `int`
        number of new clusters

    Returns
    -------
    data, df0, lda_out, mapping, titles
    '''
    if not isinstance(df0, pd.DataFrame):
        df0 = pd.read_json(df0)

    doc_list = make_doc_list(df0)
    doc_count = len(doc_list)

    print("Vectorizer Stuff", file=sys.stderr)
    tf_vectorizer, tf, tfidf, lda = create_vector_components(doc_list, n_components)

    print("Fitting Model", file=sys.stderr)
    lda.fit(tf)
    lda_out = lda.transform(tfidf)

    data = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
    mapping = data[0].topics
    print("Model fit\n", file=sys.stderr)
    titles = df0["Title"].tolist()

    lda_out_list = lda_out.tolist()
    data_json = data.to_json()

    data[1]['pos'] = add_pos_to_df(data[1], df0) #add parts of speech back into the dataframe

    doc_x = [0 for _ in range(doc_count)]
    doc_y = [0 for _ in range(doc_count)]

    # calculate coordinates for each document
    # each document has a percentage of topics in lda_out[document][topic]
    # sum product this with the coordinate of each topic in data[0]['x'][topic] and data[0]['y'][topic]
    for doc in range(doc_count):
        for topic in range(n_components):
            doc_x[doc] += lda_out_list[doc][topic] * data[0]['x'][topic]
            doc_y[doc] += lda_out_list[doc][topic] * data[0]['y'][topic]

    doc_xy={'x':doc_x, 'y':doc_y, 'titles':titles}

    data_json = data_json[:-1]+", \"mdsdoc\":"+json.dumps(doc_xy)+"}"

    #terms_to_csv(data[1], n_components) #uncomment this to save the salient terms in csv's

    return data_json, df0.to_json(), json.dumps(lda_out_list), mapping.to_json(), titles


def create_cluster(input_file, n_components):
    '''
    Creates the data of the initial clustering

    Parameters
    ----------
    n_components: `int`
        Number of desired clusters
    input_file:  `str`
        text file of abstracts with titles - MUST match format specified in create_df0

    Returns
    -------
    data, df0, lda_out, mapping, titles
    '''

    np.random.seed(11)

    print("Pre-processing...", file=sys.stderr)
    df0 = create_df0(input_file)
    print("Pre-processing complete!", file=sys.stderr)

    return cluster(df0, n_components)

def recluster(df0, n_components):
    '''
    Reclusters given data to have a new set of clusters

    Parameters
    ----------
    df0 : `DataFrame` or `json`
        generated by create_cluster. Can be either in dataframe format or JSON serialized
    n_components : `int`
        number of new clusters

    Returns
    -------
    data, df0, lda_out, mapping, titles
    '''
    if not isinstance(df0, pd.DataFrame):
        df0 = pd.read_json(df0)

    return cluster(df0, n_components)

def subset(df0, prepared_data, lda_out, n_components, t_list):
    '''
    Creates a subset of given data with user defined set of clusters
    Parameters
    ----------
    df0 : `DataFrame`
        previously generated dataframe
    prepared_data : type
        topics of prepared data (data[0] in create_cluster)
    lda_out : type
        result of lda.transform operation in create_cluster
    n_components : `int`
        number of new clusters
    t_list : type
        list of selected topics

    Returns
    -------
    data, df0, lda_out, mapping, titles
    '''
    take = 3

    df0 = pd.read_json(df0)
    lda_out = json.loads(lda_out)
    lda_out = np.array(lda_out)

    mapping = pd.read_json(prepared_data, typ='series')

    t_split = revised_topic_splitter(t_list, mapping)
    lda_model_list = retrieve_topic_docs(t_split, lda_out, take)  # list of indexes
    df_subset, _ = revised_subsetter(lda_model_list, df0)  # sent as tuple
    data_json, df0_json, lda_out_json, mapping, titles = recluster(df_subset, n_components)

    return data_json, df0_json, lda_out_json, mapping, titles

def boomerang_analyzer(doc_list):
    # we've already done all the work the analyzer is supposed to do, so just return what was passed in
    return doc_list

def create_vector_components(doc_list, n_components):
    '''
    words

    Parameters
    ----------
    doc_list : `list` [`str`]

    n_components : `int`
        number of clusters

    Returns
    -------
    tf_vectorizer, tf, tfidf, lda
    '''
    max_df = 0.8
    min_df = 0.001

    # max_features=None: Features are the words in all docs.  None uses everything
    tf_vectorizer = CountVectorizer(analyzer=boomerang_analyzer,
                                    max_df=max_df,
                                    min_df=min_df,
                                    max_features=None)
    tf = tf_vectorizer.fit_transform(doc_list)
    tfidf_transformer = TfidfTransformer(norm='l2',
                                         smooth_idf=True)
    tfidf = tfidf_transformer.fit_transform(tf)
    lda = LatentDirichletAllocation(n_components=n_components,
                                    doc_topic_prior=0.1,
                                    max_iter=15,
                                    learning_method='batch',
                                    learning_offset=50.0,
                                    n_jobs=-1,
                                    random_state=None)
    return tf_vectorizer, tf, tfidf, lda


def revised_topic_splitter(selections, mapping):
    '''
        Take user chosen list of topics and split into lists

        send to retrieve_topic_docs
    '''
    model = []
    for i in selections:
            model.append((mapping[mapping == i].index)[0])
    return model


def get_tag_from_df0(v, n, vd, term):
    '''
    Returns the part of speech tag for the given term

    Parameters
    ----------
    v, n, vd : `list` [`str`]
        Lists of terms by part of speech from get_master_pos_lists
    term : `str`
        The term whose pos is to be searched for

    Returns
    -------
    pos : `str`
        The part of speech of the term. Could be 'v', 'n', or 'vd'
    '''
    if term in v:
        return 'v'
    elif term in n:
        return 'n'
    else:
        assert(term in vd) #if it isn't v, or n  then it should be vd
        return 'vd'

def get_master_pos_lists(df0):
    '''
    Get lists of each part of speech in all documents in the DataFrame

    Parameters
    ----------
    df0 : `DataFrame`
        Should contain lists of verbs, nouns, adjectives, and verb direct object pairs for a set of documents

    Returns
    -------
    v, n,  vd : `list` [`str`]
        the master list for each part of speech
    '''
    v = []
    n = []
    vd = []

    for i in range(len(df0)):
        v.extend(df0['v'][i])
        n.extend(df0['n'][i])
        vd.extend(df0['vd'][i])

    return v, n, vd

def add_pos_to_df(terms, df0):
    '''
    Adds a part of speech column to the terms DataFrame

    Parameters
    ----------
    terms : `DataFrame`
        Contains each term
    df0 : `DataFrame`
        Contains lists of verbs, nouns, and verb direct object pairs for a set of documents

    Returns
    -------
    `Series`
        the part of speech for each term
    '''
    v, n, vd = get_master_pos_lists(df0)
    return terms.apply(lambda x: get_tag_from_df0(v, n, vd, x["Term"]), axis=1)


def terms_to_csv(terms, n_components):
    '''
    Makes a csv for each pos for each topic cluster containing terms and frequency in cluster

    Parameters
    ----------
    terms : `DataFrame`
        Contains each term, its topic, and frequency in the topic
    n_components : `int`
        number of clusters

    '''
    terms = terms.sort_values(by=['Freq'], ascending=False)

    for pos, df in terms.groupby('pos'):
        for i in range(n_components):
            try:
                df[["Term", "Freq"]].loc[df['Category'] == "Topic"+str(i+1)].to_csv(pos+str(i+1)+".csv", index=False)
            except PermissionError:
                print(pos+str(i+1)+".csv is already open")
