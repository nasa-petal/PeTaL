from flask import Markup
import re
import io


def convert(bird_data):
    """

    :param tertiaryTerm: file name of target data
    :return: Returns a list of abstracts in form [title, abstract text, source link]

    Expects target data files to exist and be in the format of:
    count\n
    title\n
    abstract text \n\n
    source link \n\n\n
    """
    for abstract in bird_data:
        cleanr = re.compile('<.*?>')  # removes HTML tags
        cleantext = re.sub(cleanr, '', abstract[3])
        abstract[3] = cleantext

    result_list = list()
    
    for abstract in bird_data:
        result_list.append(abstract[2:4])
    
    result_list = ["\r\n".join(x) for x in result_list]
    result_string = "\r\n\r\n\r\n".join(result_list)

#################Text####################33
# file_name = 'C:/Users/kkojima1/petal/petal/bird/Process.txt'
# result_list = list()

# try:
#     with open(file_name, "r", encoding="utf8") as reader:  # open the file for reading
#         text = reader.read()
# except Exception as e:  # todo: figure out error handling
#     print("Error opening file: " + str(e))

    #cleanr = re.compile('<.*?>')  # removes HTML tags
    #cleantext = re.sub(cleanr, '', bird_data)

    

    #final = cleantext.strip()
    # abstract_list = [y.strip() for y in cleantext.split(sep=u'\n\n')]  # split abstracts apart
    # del abstract_list[-1]  # remove empty entry at the end

    # result_string = ''
    # for abstract in abstract_list:
    #     single_abstract = abstract.split('\n')

    #     result_string += single_abstract[2] + '\n' + single_abstract[3] + '\n' + '\n'+ '\n'

    return result_string
    # with io.open("C:/Users/kkojima1/petal/petal/data/cluster/bird.txt", "w", encoding="utf-8") as f:
    # f.write(result_string)
