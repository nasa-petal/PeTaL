import pickle
import pandas as pd

ignore = {}

ignore['vd'] = set(pd.read_csv('verb-dobj.csv').full.values)
ignore['mn'] = set(pd.read_csv('modded-noun.csv').full.values)

f = open("verbs.txt", "r")
ignore_verbs = f.read().split(" ")
del ignore_verbs[0]
ignore['v'] = set(ignore_verbs)

f = open("other.txt", "r")
ignore_other = f.read().split(" ")
del ignore_other[0]
ignore['o'] = set(ignore_other)
ignore['o'].add('’s')
ignore['o'].add('°')
ignore['o'].add('–')
ignore['o'].add('-')
ignore['o'].add('’')

f = open("noun.txt", "r")
ignore_noun = f.read().split(" ")
del ignore_noun[0]
ignore['n'] = set(ignore_noun)
ignore['n'] = ignore['n'].union(ignore['o'])

pickle.dump(ignore, open( "ignore.p", "wb" ) )