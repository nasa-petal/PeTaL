from .module import Module

class TemplateModule(Module):
    def __init__(self, in_label='neo4j Label', out_label='neo4j Label', connect_labels=('neo4j Link Label in', 'neo4j Link Label out')):
        Module.__init__(self, in_label, out_label, connect_labels)

    def process(self, node):
        # Return single dict, list of dicts, or list of tuples.
        # If single dict or list of dicts, they are used for properties of [out_label] nodes, with connections and in_label defined by class (above)
        # If a list of tuples, it is used as (in_label, out_label, from_label, to_label, properties) to create neo4j entries, useful if a scraper returns multiple nodes or collections, especially if they are of different types
        return dict()
