"""
This module is responsible for constructing the graph structure for PeTaL

authors:
Jonathan Dowdall - jonathan.dowdall@nasa.gov
Bishoy Boktor - bishoy.boktor@nasa.gov

The graph contains nodes which represent the following:

Ontology Class (e.g. OWLClass000521):
    Node Attributes:
        label (str): Class label (e.g. morphology)
        keywords (list of str): Related keywords
        description (str): Description of class
        img (str): Image file name (img should exist in /static/)
    Edges Attributes:
        parents (classID): Identifies parent on parent-child edges
                        None if edge represents lateral connection
Literature:
    Node Attributes:
        label (str): Name of Paper/abstract
    Edge Attributes:
        literature: Identifies node as a literature node when searching neighbors
"""
import networkx as nx
from networkx.readwrite import json_graph

import pandas as pd
import petal.ontology
import json
import ontospy
import rdflib
import os

from petal import wiki_tools

def to_ont(G, label):
    ont_class = wiki_tools.get_phylum(label)
    print(label)
    print(ont_class)
    if not ont_class:
        ont_class = wiki_tools.get_kingdom(label)
    print(ont_class)
    if not ont_class:
        ont_class = label.lower()
    for c in G.node:
        if G.node[c] and G.node[c]['label'] == ont_class:
            return c
    return ""

def clean(uri):
    """ Strip class name from uri """
    return uri.split('/')[-1]

def ontospy_to_dict(c):
    """ Parses ontospy class into dictionary """
    uri = clean(c.uri.toPython())
    label = c.bestLabel().toPython()
    keywords = [clean(i.bestLabel().toPython()) for i in c.instances]
    see_also = [clean(i.toPython()) for i in c.getValuesForProperty(rdflib.RDFS.seeAlso)]
    parents = [clean(i.uri) for i in c.parents()]
    children = [clean(i.uri) for i in c.children()]
    return  {'uri' : uri,
            'label' : label,
            'keywords' : keywords,
            'see_also' : see_also,
            'parents' : parents,
            'children' : children}

def owl_to_dictionary(owl_path):
    """ Reads owl file and builds classes dictionary with ontospy """
    classes = {}
    onto = ontospy.Ontospy(owl_path)
    for c in onto.all_classes:
        ont_class = ontospy_to_dict(c)
        classes[ont_class['uri']] = ont_class
    
    return classes

def graph_to_csv(G, outpath):
    """ Writes label, img, description attributes to a csv.
    
    Args:
        G: NetworkX graph.
        outpath: path to write csv to. 
    """
    df = pd.DataFrame(dict(G.node)).transpose()
    df.drop('keywords',1).to_csv(os.path.join(os.getcwd(),outpath))

def owl_to_graph(owl_path):
    """Builds network x graph out of the owl file

    Args:
        owl_path: Path to owl ontology file.
    Returns:
        NetworkX graph of ontology.
    """
    # Owl ontology as a dictionary
    onto = owl_to_dictionary(owl_path)
    # NetworkX graph to be populated
    G = nx.Graph()
    for c in onto:
        c_data = onto[c]
        G.add_node(c, 
                    label=c_data['label'], 
                    keywords=c_data['keywords'],
                    description="Missing description...",
                    img='{}.jpg'.format(c_data['label']))
        # Build edges
        # Identify parent as edge attribute since (we do not want to use directed graph)
        for p in c_data['parents']:
            G.add_edge(p,c, parents=p)
        for r in c_data['see_also']:
            G.add_edge(c,r, parents=None)
    return G

def csv_to_graph(G, data_path):
    """Updates network x graph from csv stored data

    Args:
        G: NetworkX graph.
        data_path: Path to csv with stored graph data.
    Returns:
        Graph updated with csv data.
    """
    # Just to be safe we don't modify graph in place.
    G_copy = G.copy()
    # Data from csv store - used for images and descriptions
    df = pd.read_csv(data_path, index_col=0)
    for c in G_copy:
        G_copy.node[c].update(df.loc[c])
    return G_copy

def lit_to_graph(G, lit_path):
    """Updates network x graph with literature data

    Arg:
        G: NetworkX graph
        lit_path: Path to csv containing literature data

    Returns:
        Graph updated with literature nodes
    """

    G_copy = G.copy()

    df = pd.read_csv(lit_path)
    df['environment'] = df['environment'].str.split(';')
    df['morphology'] = df['morphology'].str.split(';')
    df.fillna(' ')

    # Iterating through df row-wise
    for row in df.itertuples(index = True, name = 'Pandas'):
        current_row_title = getattr(row, 'title').rstrip()
        current_row_environment = getattr(row, 'environment')
        current_row_morphology = getattr(row, 'morphology')

        # Creating node in graph out of the lit title
        G_copy.add_node(current_row_title, label=current_row_title)

        # Building edges between nodes if they are list and not NaN
        if type(current_row_environment) == list:
            for env_item in current_row_environment:
                G_copy.add_edge(current_row_title, env_item.strip(), literature='literature')
        
        if type(current_row_morphology) == list:
            for morph_item in current_row_morphology:
                G_copy.add_edge(current_row_title, morph_item.strip(), literature='literature')

    return G_copy

def build_graph(owl_path='ontology.owl', data_path='classes.csv', lit_path='literature.csv'):
    """Construct networkx graph with OWL and CSV data

    Args:
        owl_path: Relative path to OWL ontology file.
        data_path: Relative path to CSV data. If file does not exist,
            one will be written to this path.
        lit_path: Relative path to literature CSV data. 

    Returns:
        NetworkX graph of data. 
    """
    # Consruct graph from owl file
    G = owl_to_graph(owl_path)
    # Supplement data from CSV if exists
    if os.path.isfile(data_path):
        G = csv_to_graph(G, data_path)
    # Else create CSV
    else:
        graph_to_csv(G, data_path)

    # Supplement data from Literature CSV
    if os.path.isfile(lit_path):
        G = lit_to_graph(G, lit_path)
    return G

def build_graph_json(G):
    graph_data = json_graph.node_link_data(G)
    with open('petal/data/graph.json', 'w') as outfile:
        json.dump(graph_data, outfile)

def get_lit(G, ont_class):
    """ Recursively get literature from node and children. """
    lit = []
    for neighbor,edge in G[ont_class].items():
        if 'literature' in edge:
            lit.append(neighbor.strip())
            continue
        if edge['parents'] == ont_class:
            lit += get_lit(G, neighbor)
    return lit

def create_profile(G, ont_class, debug=False, data_path=None, lit_path=None):
    """ Packages node attributes and edges into dictionary 

    Collects labels and relationships of neighbors to 
    encapsulate a single class profile.
    Args:
        G: NetworkX graph.
        ont_class: ClassID to generate profle for
        debug (optional): Checks for updates to CSV file.
        data_path (optional): CSV with class info, required if debug = True
        lit_path (optional): CSV with literature info, required if debug = True
    Returns:
        Dictionary of node data to be sent to front-end templates
    """
    # If debugging, reload description/img upon every profile construction
    #if debug:
        #G = csv_to_graph(G, data_path)
        #G = lit_to_graph(G, lit_path)
    # Copy node so we don't modify graph
    profile = G.nodes()[ont_class]
    profile['parents'] = {}
    profile['children'] = {}
    profile['see_also'] = {}
    profile['literature'] = get_lit(G, ont_class)
    # Add edges, check for parent-child, see-also relationship, or literature connection
    for neighbor,edge in G[ont_class].items():
        if 'literature' in edge:
            continue
        label = G.node[neighbor]['label']
        relation = 'see_also'
        if edge['parents'] == neighbor:
            relation = 'parents'
        elif edge['parents'] == ont_class:
            relation = 'children'
        profile[relation][neighbor] = label
    return profile


