
def neo_add_json(tx, label='Generic', properties=None):
    prop_set = '{' + ','.join('{key}:{{{key}}}'.format(key=k) for k in properties) + '}'
    query = 'CREATE (n:{label} '.format(label=label) + prop_set + ')'
    return tx.run(query, **properties)
