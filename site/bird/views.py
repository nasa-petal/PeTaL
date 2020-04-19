from django.shortcuts import render
from django.http import HttpResponse

from neo4j import GraphDatabase, basic_auth

from time import time

from .search import search

def index(request):
    context = dict(query='')
    return render(request, 'bird_e2b.html', context)

def results(request):
    query = request.GET.get('q')
    duration, results = search(query)

    articles = []

    start = time()
    neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"), encrypted=False)
    with neo_client.session() as session:
        fetch_start = time()
        for term_results in results:
            for a, b, uuid in term_results:
                article = session.run('MATCH (a:Article) WHERE a.uuid = \'{uuid}\' RETURN a'.format(uuid=uuid))
                article = next(article.records())['a']
                articles.append(dict(title=article['title'], abstract=article['summary'], authors='', relevancy=str(a) + ' ' + str(b)))
        fetch_end = time()
    end = time()
    context = dict(search_time=round(duration, 6), load_time=round(end-start, 6), fetch_time=round(fetch_end - fetch_start, 6), papers=articles)
    return render(request, 'bird_results.html', context)
