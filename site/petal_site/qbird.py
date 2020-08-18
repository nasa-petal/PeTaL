import spacy, re
import pandas as pd
from nltk.corpus import wordnet as wn
import json

def load_dict(path):
    '''
    Loads a thesaurs and returns the engineer and biologist terms as a dictionary 
    
    Parameters
    ----------
    path: `str`
        The path to the thesarus

    Returns
    -------
    terms: `dict`
        Dictionary with engineering terms as keys and lists of biologist terms as values
    '''

    with open(path) as st:
        data = json.load(st)  # open the json containing the thesaurus

    terms={}

    try:
        for tert in data["tertiary"]:
            terms[tert["engineer"]] = tert["biologist"]
    except KeyError as e:
        print(e.args[0], " is not in the thesaurs")
        raise


    return terms

def load_nlp():
    '''
    Tries to return loaded spacy nlp or errors
    '''
    
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        print("\n ERROR: Cannot find spacy model. Try 'python -m spacy download en' \n")
        raise


def get_synonyms(term, pos, target_pos=None):
    '''
    Get synonyms of a term with the same part of speech or the target pos. Heavily inspired by https://stackoverflow.com/a/16752477

    Parameters
    ----------
    term: `str`
        The term we want synonyms of
    pos: `str`
        The part of speech of the term
    target_pos: `str` or `None`
        If we want the synonyms to be of a different part of speech than the term, this will be the pos we want

    Returns
    -------
    `set`
        The set of synonyms
    '''

    synsets = wn.synsets(term, pos=pos)
    
    # Word not found
    if not synsets:
        print("Word not found:", term, "with pos \'"+pos+"\'")
        return []

    # Get all verb lemmas of the word
    lemmas = [l for s in synsets for l in s.lemmas() if s.name().split('.')[1] == pos]

    if target_pos:
        # Get related forms
        derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

        # filter only the target pos
        related_lemmas = [l for drf in derivationally_related_forms for l in drf[1] if l.synset().name().split('.')[1] == target_pos]

        # Extract the words from the lemmas
        words = {l.name().replace('_', ' ') for l in related_lemmas}

        #return words

    else:
        # Extract the words from the lemmas
        words = {l.name().replace('_', ' ') for l in lemmas}

    return words

def combos(modifiers, roots):
    '''
    Makes combinations of modifiers and roots
    
    Parameters
    ----------
    modifiers: `set` (`str`)
        The set of modifiers

    roots: `set` (`str`)
        The set of roots

    Returns
    -------
    combos: `set` (`str`)
        Set of modifier + root combinations
    ''' 

    combos = set([])
    for mod in modifiers:
        for root in roots:
            combos.add(mod + " " + root)
            combos.add(mod)
            combos.add(root)
            
    return combos

def get_clean_pos(pos):
    '''
    Returns spacy part of speech as nltk likes them
    
    Parameters
    ----------
    pos: `str`
        A part of speech like 'NOUN'

    Returns
    -------
    `str`:
        A part of speech like 'n'
    ''' 
    return pos[0].lower()

def get_lemma(term):
    return term.lemma_

def get_potential_matches(term, term_token, potential_exact_matches, potential_inexact_matches):
    '''
    Returns two sets of potential matches
    
    Parameters
    ----------
    term: `str`
        The current term. May be made up of more than one token
    term_token: `spacy.tokens.token.Token`
        The current spacy token
    potential_exact_matches: `set` (`str`)
        The current set of potential exact matches
    potential_inexact_matches: `set` (`str`)
        The current set of potential inexact matches

    Returns
    -------
    potential_exact_matches: `set` (`str`)
    potential_inexact_matches: `set` (`str`)
    ''' 

    term_token_lemma = get_lemma(term_token)
    term_token_head_lemma = get_lemma(term_token.head)
    term_token_head_head_lemma = get_lemma(term_token.head.head)

    if (term_token.dep_ == "compound" or (term_token.pos_ == "ADJ" and term_token.dep_ != "ROOT")):
        potential_exact_matches.add(term_token_lemma)

        term_syns = get_synonyms(term, get_clean_pos(term_token.pos_))
        head_syns = get_synonyms(term_token_head_lemma, get_clean_pos(term_token.head.pos_))
        head_head_syns = get_synonyms(term_token_head_head_lemma, get_clean_pos(term_token.head.head.pos_))

        if term_token.pos_ == "ADJ":
            # e.g. red and green trees (or just red trees) -> red trees (and green trees)        
            potential_exact_matches.add(term_token_lemma + " " + term_token_head_head_lemma)
            potential_inexact_matches = potential_inexact_matches.union(combos(term_syns, head_head_syns))

        else:
            # e.g. friction and drag reduction (or just friction reduction) -> friction reduction
            potential_exact_matches.add(term_token_lemma + " " + term_token_head_lemma)
            potential_inexact_matches = potential_inexact_matches.union(combos(term_syns, head_syns))

            if term_token_head_lemma != term_token_head_head_lemma:
                # e.g. friction and drag reduction -> drag reduction
                potential_exact_matches.add(term_token_head_head_lemma + " " + term_token_head_lemma)
                potential_inexact_matches = potential_inexact_matches.union(combos(head_head_syns, head_syns))


    elif term_token.dep_ == "dobj" and term_token.head.pos_=="VERB": 
        if "aux" not in term_token.head.dep_ :
            nouns = get_synonyms(term_token_head_lemma, 'v', 'n') # nominalize the verb (e.g. reduce -> reduction)
            dobjs = get_synonyms(term_token_lemma, 'n') # get synonyms of direct object
            
            potential_exact_matches.add(term_token_lemma)
            potential_inexact_matches = potential_inexact_matches.union(combos(dobjs, nouns))

    else:
        if (term_token.pos_=="NOUN" or term_token.pos_=="PROPN"):
            potential_exact_matches.add(term_token_lemma)
            potential_inexact_matches = potential_inexact_matches.union(get_synonyms(term_token_lemma, 'n'))
        elif term_token.pos_=="VERB" and "aux" not in term_token.dep_:
            potential_exact_matches.add(term_token_lemma)
            potential_inexact_matches = potential_inexact_matches.union(get_synonyms(term_token_lemma, 'v'))

    return potential_exact_matches, potential_inexact_matches

def get_eng_terms(query, terms, nlp):
    '''
    Returns the engineering term matches in the query
    
    Parameters
    ----------
    query: `str`
        The query that is being searched through
    terms: `dict`
        The thesaurs that contains the engineering terms
    nlp: `spacy.lang.en.English`
        The spacy nlp loaded in from `load_nlp`

    Returns
    -------
    exact_matches: `set` (`str`)
        The engineering terms that match with a term in the query exactaly
    exact_synonym_matches: `set` (`str`)
        The engineering term matches with synonyms
    partial_matches: `set` (`str`)
        The engineering terms that contain a potential match as well as some other words
    ''' 
    parsed_text = nlp(query)
    
    potential_exact_matches=set([])
    potential_inexact_matches=set([])
    
    for term_token in parsed_text:
        exact, inexact = get_potential_matches(term_token.lemma_, term_token, potential_exact_matches, potential_inexact_matches)
        potential_exact_matches = potential_exact_matches.union(exact)
        potential_inexact_matches = potential_inexact_matches.union(inexact)

    potential_inexact_matches = potential_inexact_matches.union(potential_exact_matches)
    
    exact_matches = set([])
    exact_synonym_matches = set([])
    partial_matches = set([])

    for search_term in potential_exact_matches:
        for eng_term in terms:
            if search_term == eng_term:
                exact_matches.add(eng_term)
    
    for search_term in potential_inexact_matches:
        for eng_term in terms:
            if eng_term not in exact_matches and search_term == eng_term:
                exact_synonym_matches.add(eng_term)
            elif eng_term not in exact_matches and eng_term not in exact_synonym_matches and re.search(r"\b" + re.escape(search_term) + r"\b", eng_term):
                partial_matches.add(eng_term)
                
    return exact_matches, exact_synonym_matches, partial_matches

def print_results(queries, terms, nlp):
    for question in queries:
        print("Query:", question)
        exact_matches, exact_synonym_matches, partial_matches = get_eng_terms(question, terms, nlp)
        print("Exact Matches:", exact_matches)
        print("Exact Synonym Matches:", exact_synonym_matches)
        print("Partial Matches:", partial_matches)
        print()


def test(): 
    queries = ["drag and friction reduction","motion perception", "how to perceive motion", "sense movement", "how to sense movement"]
    terms = load_dict("static/js/NTRS_data.js")
    nlp = load_nlp()
    print_results(queries, terms, nlp)

def process_with_nlp(query): 
    terms = load_dict("static/js/NTRS_data.js")
    nlp = load_nlp()
    exact_matches, exact_synonym_matches, partial_matches = get_eng_terms(query, terms, nlp)
    return exact_matches, exact_synonym_matches, partial_matches

if __name__ == "__main__":
    test()    
