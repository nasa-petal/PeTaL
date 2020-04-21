from django.shortcuts import render
from django.http import HttpResponse

from neo4j import GraphDatabase, basic_auth

from time import time

from .search import search

# TODO move the client elsewhere? This is just for optimization/testing - Lucas
neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"), encrypted=False)
session = neo_client.session()

def index(request):
    context = dict(query='')
    return render(request, 'bird_e2b.html', context)

def results(request):
    query = request.GET.get('q')
    duration, results = search(query)

    articles = []

    fetch_start = time()
    for term_results in results:
        for a, b, uuid in term_results:
            article = session.run('MATCH (a:Article) WHERE a.uuid = \'{uuid}\' RETURN a'.format(uuid=uuid))
            article = next(article.records())['a']
            articles.append(dict(title=article['title'], abstract=article['summary'], authors='', relevancy=str(a) + ' ' + str(b)))
    fetch_end = time()
    context = dict(search_time=round(duration, 10), fetch_time=round(fetch_end - fetch_start, 6), papers=articles)
    if len(articles) > 0:
        return render(request, 'bird_results.html', context)
    else:
        return render(request, 'bird_no_results.html', context)
