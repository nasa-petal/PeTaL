from petal.pipeline.module_utils.module import Module

class MockEOLSpeciesModule(Module):
    def __init__(self, in_label=None, out_label='EOLPage', connect_labels=None, name='EOLSpecies'):
        Module.__init__(self, in_label, out_label, connect_labels, name, page_batches=False)

    def process(self, driver=None):
        taxon_properties = {'kingdom'     : 'Animalia', 
                            'phylum'      : 'Chordata', 
                            'class'       : 'Vertebrata', 
                            'order'       : 'Mammals', 
                            'superfamily' : '', 
                            'family'      : 'Rorquals', 
                            'genus'       : 'Megaptera', 
                            'subgenus'    : '', 
                            'species'     : 'Megaptera novaeangliae',
                            'name'        : 'Megaptera novaeangliae'}
        name = taxon_properties['species']
        yield self.custom_transaction(data=taxon_properties, in_label=None, out_label='Taxon', uuid=name, connect_labels=None)
        data = {'canonical' : name, 'page_id' : 46559443, 'rank' : 'species'}
        yield self.custom_transaction(data=data, in_label='Taxon', out_label='EOLPage', uuid=str(data['page_id']), from_uuid=name, connect_labels=('eol_page', 'eol_page'))
        taxon_properties = {'kingdom'     : 'Animalia', 
                            'phylum'      : 'Chordata', 
                            'class'       : 'Aves', 
                            'order'       : 'Falconiformes', 
                            'superfamily' : '', 
                            'family'      : 'Falconidae', 
                            'genus'       : 'Falco', 
                            'subgenus'    : '', 
                            'species'     : 'Falco peregrinus',
                            'name'        : 'Falco peregrinus'}
        name = taxon_properties['species']
        yield self.custom_transaction(data=taxon_properties, in_label=None, out_label='Taxon', uuid=name, connect_labels=None)
        data = {'canonical' : name, 'page_id' : 45510796, 'rank' : 'species'}
        yield self.custom_transaction(data=data, in_label='Taxon', out_label='EOLPage', uuid=str(data['page_id']), from_uuid=name, connect_labels=('eol_page', 'eol_page'))

        taxon_properties = {'kingdom'     : 'Animalia', 
                            'phylum'      : 'Chordata', 
                            'class'       : 'Mammalia', 
                            'order'       : 'Chiroptera', 
                            'superfamily' : '', 
                            'family'      : 'Phyllostomidae', 
                            'genus'       : 'Glossophaga', 
                            'subgenus'    : '', 
                            'species'     : 'Glossophaga soricina',
                            'name'        : 'Glossophaga soricina'}
        name = taxon_properties['species']
        yield self.custom_transaction(data=taxon_properties, in_label=None, out_label='Taxon', uuid=name, connect_labels=None)
        data = {'canonical' : name, 'page_id' : 327431, 'rank' : 'species'}
        yield self.custom_transaction(data=data, in_label='Taxon', out_label='EOLPage', uuid=str(data['page_id']), from_uuid=name, connect_labels=('eol_page', 'eol_page'))

        taxon_properties = {'kingdom'     : 'Animalia', 
                            'phylum'      : 'Chordata', 
                            'class'       : 'Chondrichthyes', 
                            'order'       : 'Lamniformes', 
                            'superfamily' : '', 
                            'family'      : 'Lamnidae', 
                            'genus'       : 'Carcharodon', 
                            'subgenus'    : '', 
                            'species'     : 'Carcharodon carcharias',
                            'name'        : 'Carcharodon carcharias'}
        name = taxon_properties['species']
        yield self.custom_transaction(data=taxon_properties, in_label=None, out_label='Taxon', uuid=name, connect_labels=None)
        data = {'canonical' : name, 'page_id' : 46559751, 'rank' : 'species'}
        yield self.custom_transaction(data=data, in_label='Taxon', out_label='EOLPage', uuid=str(data['page_id']), from_uuid=name, connect_labels=('eol_page', 'eol_page'))

        taxon_properties = {'kingdom'     : 'Animalia', 
                            'phylum'      : 'Chordata', 
                            'class'       : 'Aves', 
                            'order'       : 'Accipitriformes', 
                            'superfamily' : '', 
                            'family'      : 'Accipitridae', 
                            'genus'       : 'Haliaeetus', 
                            'subgenus'    : '', 
                            'species'     : 'Haliaeetus leucocephalus',
                            'name'        : 'Haliaeetus leucocephalus'}
        name = taxon_properties['species']
        yield self.custom_transaction(data=taxon_properties, in_label=None, out_label='Taxon', uuid=name, connect_labels=None)
        data = {'canonical' : name, 'page_id' : 45511401, 'rank' : 'species'}
        yield self.custom_transaction(data=data, in_label='Taxon', out_label='EOLPage', uuid=str(data['page_id']), from_uuid=name, connect_labels=('eol_page', 'eol_page'))

        taxon_properties = {'kingdom'     : 'Animalia', 
                            'phylum'      : 'Chordata', 
                            'class'       : 'Aves', 
                            'order'       : 'Accipitriformes', 
                            'superfamily' : '', 
                            'family'      : 'Accipitridae', 
                            'genus'       : 'Haliaeetus', 
                            'subgenus'    : '', 
                            'species'     : 'Haliaeetus leucocephalus',
                            'name'        : 'Haliaeetus leucocephalus'}
        name = taxon_properties['species']
        yield self.custom_transaction(data=taxon_properties, in_label=None, out_label='Taxon', uuid=name, connect_labels=None)
        data = {'canonical' : name, 'page_id' : 45511401, 'rank' : 'species'}
        yield self.custom_transaction(data=data, in_label='Taxon', out_label='EOLPage', uuid=str(data['page_id']), from_uuid=name, connect_labels=('eol_page', 'eol_page'))

        taxon_properties = {'kingdom'     : 'Animalia', 
                            'phylum'      : 'Chordata', 
                            'class'       : 'Actinopterygii', 
                            'order'       : 'Scombriformes', 
                            'superfamily' : '', 
                            'family'      : 'Sphyraenidae', 
                            'genus'       : 'Sphyraena', 
                            'subgenus'    : '', 
                            'species'     : '',
                            'name'        : 'Sphyraena'}
        name = taxon_properties['genus']
        yield self.custom_transaction(data=taxon_properties, in_label=None, out_label='Taxon', uuid=name, connect_labels=None)
        data = {'canonical' : name, 'page_id' : 46577212, 'rank' : 'genus'}
        yield self.custom_transaction(data=data, in_label='Taxon', out_label='EOLPage', uuid=str(data['page_id']), from_uuid=name, connect_labels=('eol_page', 'eol_page'))
