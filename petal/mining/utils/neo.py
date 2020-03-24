from time import sleep

def add_json_node(tx, label='Generic', properties=None):
    if properties is None:
        properties = dict()
    prop_set = '{' + ','.join('{key}:{{{key}}}'.format(key=k) for k in properties) + '}'
    query = 'MERGE (n:{label} '.format(label=label) + prop_set + ') RETURN n'
    return tx.run(query, **properties)

def get_count(tx, finder):
    # For instance:
    # finder = 'MATCH (n:Article) WHERE n.journal CONTAINS 'Experimental Biology'
    count_query = finder + ' WITH COUNT (n) AS count RETURN count LIMIT 1'
    count       = tx.run(count_query)
    count       = list(count.records())[0]['count']
    return count

def get_page_queries(query, count, page_size=1000, rate_limit=0.25):
    properties = dict()
    for i in range(count // page_size):
        properties['skip']  = i * page_size
        properties['limit'] = page_size
        page_query = query + ' SKIP {skip} LIMIT {limit}'
        yield page_query, properties
        sleep(rate_limit)

def page(tx, finder, query, properties=None, page_size=1000, rate_limit=0.25):
    if properties is None:
        properties = dict()
    # For instance:
    # query  = MATCH (n:Article) WHERE n.abstract CONTAINS 'Aerodynamics'
    # finder = 'MATCH (n:Article) WHERE n.journal CONTAINS 'Experimental Biology'
    # WITH COUNT (n) AS count RETURN count LIMIT 1'
    count = get_count(tx, finder)
    for query, q_properties in get_page_queries(query, count, page_size=page_size, rate_limit=rate_limit):
        properties.update(q_properties)
        yield tx.run(query, **properties)
