
def add_json(tx, label='Generic', properties=None):
    if properties is None:
        properties = dict()
    prop_set = '{' + ','.join('{key}:{{{key}}}'.format(key=k) for k in properties) + '}'
    query = 'CREATE (n:{label} '.format(label=label) + prop_set + ')'
    return tx.run(query, **properties)

def page(tx, finder, query, properties=None, page_size=1000):
    if properties is None:
        properties = dict()
    # For instance:
    # query  = MATCH (n:Article) WHERE n.abstract CONTAINS 'Aerodynamics'
    # finder = 'MATCH (n:Article) WHERE n.journal CONTAINS 'Experimental Biology'
    # WITH COUNT (n) AS count RETURN count LIMIT 1'
    count_query = finder + ' WITH COUNT (n) AS count RETURN count LIMIT 1'
    count       = tx.run(count_query, **properties)
    count       = list(count.records())[0]['count']

    for i in range(count // page_size):
        properties['skip']  = i * page_size
        properties['limit'] = page_size
        page_query = query + ' SKIP {skip} LIMIT {limit}'
        yield tx.run(page_query, **properties)
