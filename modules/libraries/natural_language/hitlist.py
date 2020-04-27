import pickle

class HitList:
    def __init__(self, uuid):
        uuid = uuid.replace('\'', '').replace('(', '').replace(')', '').replace('\\', '').replace('\n', '').replace('?', '')
        self.filename = 'data/hitlists/{}.hitlist'.format(uuid)
        self.sections = dict()
        self.words    = set()
        self.uuid     = uuid

    def add(self, section, word):
        self.words.add(word)

        if section not in self.sections:
            self.sections[section] = dict()
        section_counter = self.sections[section]
        if word in section_counter:
            section_counter[word] += 1
        else:
            section_counter[word] = 1

    def word_hitlist(self, word):
        word_list = dict()
        for section, section_counter in self.sections.items():
            word_list[section] = section_counter.get(word, 0)
        return tuple(word_list.keys()), tuple(word_list.values())

    def save(self):
        with open(self.filename, 'wb') as outfile:
            pickle.dump((self.sections, self.words), outfile)

    def load(self):
        with open(self.filename, 'rb') as infile:
            self.sections, self.words = pickle.load(infile)
