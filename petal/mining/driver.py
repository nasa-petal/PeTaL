from eol  import search as eol_search
from wiki import search as wiki_search

def main():
    # TODO: read species from backbone

    species = 'Hapalochlaena lunulata'

    print(eol_search(species))
    print(wiki_search(species))

if __name__ == '__main__':
    main()
