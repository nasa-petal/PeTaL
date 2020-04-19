from django.shortcuts import render
from django.http import HttpResponse

from neo4j import GraphDatabase, basic_auth

from .search import search

def index(request):
    context = dict(query='')
    return render(request, 'bird_e2b.html', context)

def results(request):
    query = request.GET.get('q')
    results = search(query)

    articles = []

    neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"), encrypted=False)
    with neo_client.session() as session:
        for term_results in results:
            for a, b, uuid in term_results:
                article = session.run('MATCH (a:Article) WHERE a.uuid = \'{uuid}\' RETURN a'.format(uuid=uuid))
                article = next(article.records())['a']
                articles.append(article)
    # articles = [dict(title='Paper from search ' + result, authors='Authors', relevancy='100', abstract='This is the abstract of the paper')]
    context = dict(papers=articles)
    return render(request, 'bird_results.html', context)
