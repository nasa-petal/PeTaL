from ..utils.module import Module

bread_trees = '''Encephalartos aemulans
Encephalartos altensteinii
Encephalartos aplanatus
Encephalartos arenarius
Encephalartos barteri
Encephalartos brevifoliolatus 
Encephalartos bubalinus 
Encephalartos caffer 
Encephalartos cerinus 
Encephalartos chimanimaniensis 
Encephalartos concinnus 
Encephalartos cupidus 
Encephalartos cycadifolius 
Encephalartos delucanus 
Encephalartos dolomiticus 
Encephalartos dyerianus 
Encephalartos equatorialis 
Encephalartos eugene
Encephalartos ferox 
Encephalartos friderici
Encephalartos ghellinckii 
Encephalartos gratus 
Encephalartos heenanii 
Encephalartos hildebrandtii 
Encephalartos hirsutus 
Encephalartos horridus 
Encephalartos humilis 
Encephalartos inopinus 
Encephalartos ituriensis 
Encephalartos kisambo 
Encephalartos laevifolius 
Encephalartos lanatus 
Encephalartos latifrons 
Encephalartos laurentianus 
Encephalartos lebomboensis 
Encephalartos lehmannii 
Encephalartos longifolius 
Encephalartos mackenziei 
Encephalartos macrostrobilus 
Encephalartos manikensis 
Encephalartos marunguensis 
Encephalartos middelburgensis 
Encephalartos msinganus 
Encephalartos munchii 
Encephalartos natalensis 
Encephalartos ngoyanus 
Encephalartos nubimontanus 
Encephalartos paucidentatus 
Encephalartos poggei 
Encephalartos princeps 
Encephalartos pruniferus 
Encephalartos pterogonus 
Encephalartos pungens 
Encephalartos relictus 
Encephalartos repandus 
Encephalartos schaijesii 
Encephalartos schmitzii 
Encephalartos sclavoi 
Encephalartos senticosus Vorster
Encephalartos septentrionalis 
Encephalartos tegulaneus 
Encephalartos transvenosus 
Encephalartos trispinosus 
Encephalartos turneri 
Encephalartos umbeluziensis 
Encephalartos villosus 
Encephalartos whitelockii 
Encephalartos woodii'''

aloe_trees = '''
Aloe africana 
Aloe alooides 
Aloe angelica 
Aloe arborescens 
Aloe cameronii 
Aloe castanea 
Aloe comosa 
Aloe dolomitica 
Aloe excelsa 
Aloe ferox 
Aloe khamiensis 
Aloe littoralis 
Aloe marlothii 
Aloe munchii 
Aloe pearsonii 
Aloe pluridens 
Aloe pretoriensis 
Aloe rupestris 
Aloe sessiliflora 
Aloe speciosa 
Aloe spectabilis 
Aloe thraskii 
'''

class TestingModule(Module):
    '''
    This module populates neo4j with Species nodes, allowing WikipediaModule and others to process them.
    Notice how BackboneModule's in_label is None, which specifies that it is independent of other neo4j nodes
    '''
    def __init__(self, in_label=None, out_label='Species', connect_label=None, name='Testing', count=1):
        Module.__init__(self, in_label, out_label, connect_label, name, count)

    def process(self):
        for species in aloe_trees.split('\n'):
            name = species.strip()
            yield self.default_transaction({'name' : name}, uuid=name)
        for species in bread_trees.split('\n'):
            name = species.strip()
            yield self.default_transaction({'name' : name}, uuid=name)
