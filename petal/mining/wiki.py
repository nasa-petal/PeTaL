import wikipedia

def search(query):
    results = wikipedia.search(query)
    if len(results) == 0:
        raise ValueError('No wikipedia results for {}'.format(query))
    first_page = wikipedia.page(results[0])
    print(first_page)
    print(dir(first_page))
    return first_page
