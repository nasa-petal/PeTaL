
def search(tx, term):
    query = 'MATCH (n:Article) WHERE n.abstract CONTAINS {term} RETURN n'
    return tx.run(query, term=term)
