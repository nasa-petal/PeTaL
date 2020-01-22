import os
from flask import Markup

def get_papers(tertiaryTerm):
    """

    :param tertiaryTerm: file name of target data
    :return: Returns a list of abstracts in form [title, abstract text, source link]

    Expects target data files to exist and be in the format of:
    count\n
    title\n
    abstract text \n\n
    source link \n\n\n
    """

    petal_dir = os.path.abspath(os.path.dirname(__file__))
    file_name = os.path.join( petal_dir, '..', tertiaryTerm )

    # file_name = 'petal/data/bird/' + tertiaryTerm + '.txt'
    result_list = list()

    try:
        with open(file_name, "r", encoding="utf8") as reader:  # open the file for reading
            text = reader.read()
    except Exception as e:  # todo: figure out error handling
        print("Error opening file: " + str(e))
        return None

    relevancy_denom = 0
    abstract_list = [y.strip() for y in text.split(sep=u'\n\n')]  # split abstracts apart
    del abstract_list[-1]  # remove empty entry at the end

    for abstract in abstract_list:
        single_abstract = abstract.split('\n')

        if relevancy_denom == 0:    # get the count for the most relevant abstract
            relevancy_denom = int(single_abstract[0])

        single_abstract[0] = int(single_abstract[0]) * 100 // relevancy_denom
        single_abstract[3] = Markup(single_abstract[3])
        result_list.append(single_abstract)
    return result_list
