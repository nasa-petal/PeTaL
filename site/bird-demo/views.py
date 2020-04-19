from django.shortcuts import render
from django.http import HttpResponse

from neo4j import GraphDatabase, basic_auth

def index(request):
    context = dict(query='')
    return render(request, 'bird/bird_e2b.html', context)

def results(request):
    query = request.GET.get('q')
    neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"), encrypted=False)
    with neo_client.session() as session:
        result = session.run('MATCH (a:Article) WHERE a.content CONTAINS \'{test}\' RETURN a'.format(test=query))
    articles = [article['a'] for article in result.records()]
    print(str(articles).encode('utf-8'), flush=True)
    context = dict(papers=articles)
    return render(request, 'bird_results.html', context)
