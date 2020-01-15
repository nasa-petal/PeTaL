# -*- coding: utf-8 -*-
"""
Spyder Editor

Author Kei Kojima
"""

import json
import os

"""
Will create "hierarchy.json and strategies.json" from BIRD data. An error message will be 
triggered on line 99 due to data inconsistencies from the BIRD data. Due to restriction of 
time author did not find a solution. For now if parts of BIRD data are cleaned it has worked. 
"""

bird_abstracts = ["Articulate", "Actuate", "Allow-DOF", "Analog", "Change", "Collect",
                  "Composite", "Condition", "Contain", "Control", "Convert", "Couple",
                  "Decrease", "Decrement", "Detect", "Discrete", "Distribute", 
                  "Divide", "Export", "Guide", "Import", "Increase", "Indicate",
                  "Inhibit", "Join", "Link", "Measure", "Mix", "Particulate", 
                  "Prevent", "Process", "Regulate", "Remove", "Rotate", "Secure",
                  "Separate", "Shape", "Stabilize", "Stop", "Store", "Supply", "Support",
                  "Transfer", "Translate", "Transmit", "Transport"]

result_list = list()
result_dict = dict()
tertiary_term_dict = dict()

petal_dir = os.path.abspath(os.path.dirname(__file__))

for txt_file in bird_abstracts:
    current_tertiary_list = list()
    file_name = os.path.join( petal_dir, 'data/bird', txt_file + '.txt' )
    try:
        with open(file_name, "r", encoding="utf8") as reader:  # open the file for reading
            text = reader.read()
    except Exception as e:  # todo: figure out error handling
        print("Error opening file: " + str(e))
    
    abstract_list = [y.strip() for y in text.split(sep=u'\n\n')]  # split abstracts apart
    del abstract_list[-1]  # remove empty entry at the end
    for abstract in abstract_list:
        single_abstract = abstract.split('\n')
        result_list.append(single_abstract)
        current_tertiary_list.append(single_abstract[5])
    
    tertiary_term_dict[txt_file] = current_tertiary_list


####################################  (creates same database as hierarchy.json)
"""Create "abstract list" for BIRD"""
####################################
final = list()
for i in result_list:
    result_dict = dict()
    result_dict['post_id'] = i[5]
    result_dict['post_title'] = i[2]
    result_dict['permalink'] = i[4]
    final.append(result_dict)

####################################################################


file_name = os.path.join( petal_dir, 'static/js', 'bird_hierarchy.txt' )

# file_name = 'C:/Users/kkojima1/petal/petal/static/js/bird_hierarchy.txt'
with open(file_name, "r", encoding="utf8") as reader:  # open the file for reading
            data = reader.read()


data = data.replace('id:', '"value":')

data = data.replace('name', '"name"')
data = data.replace('secondary', '"children"')
data = data.replace('tertiary', '"children"')
data = data.replace('bioTerms', '"children"')

#print(data)
data = "".join(data.split())
data = data.replace('},]', '}]')
data = data.replace('},}', '}}')
data = data.replace(',}', '}')

data = data.replace('"value": 0', '"value":5')

#data = '{"name":"ROOT","value":1774,"parent":"null","children":' + data + '}'
python_dict = json.loads(data)

for secondary1 in python_dict:
    #print(secondary1)
    print(secondary1)
    print('///////////////')
    for secondary2 in secondary1["children"]:
    
        for tertiary1 in secondary2["children"]:
            tertiary1["children"] = tertiary_term_dict[str(tertiary1["name"])]
python_dict = {"name":"ROOT","value":0,"parent":"null","children":python_dict}


#######################
    
test_list = final
  
# using naive method to  
# remove duplicates  
res_list = []
for i in range(len(test_list)): 
    if test_list[i] not in test_list[i + 1:]: 
        res_list.append(test_list[i])

#################################

file_name = os.path.join( petal_dir, 'static/js', 'strategies1.json' )

# From Kei
# with open('petal/static/js/strategies1.json', 'w') as outfile:




# with open('C:/Users/kkojima1/petal/petal/static/js/strategies1.json', 'w') as outfile:
with open(file_name, 'w') as outfile:
    json.dump(res_list, outfile)

# From Kei
# with open('petal/static/js/hierarchy1.json', 'w') as outfile:

    
file_name = os.path.join( petal_dir, 'static/js', 'hierarchy1.json' )
# with open('C:/Users/kkojima1/petal/petal/static/js/hierarchy1.json', 'w') as outfile:
with open(file_name, 'w') as outfile:
    json.dump(python_dict, outfile)
