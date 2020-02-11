from django.shortcuts import render
from django.http import HttpResponse

from neo4j import GraphDatabase, basic_auth

def index(request):
    context = dict(query='')
    return render(request, 'bird/bird_e2b.html', context)

def results(request):
    query = request.GET.get('q')
    # neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"))
    neo_client = GraphDatabase.driver("bolt://139.88.179.199:7667", auth=basic_auth("neo4j", "testing"))
    with neo_client.session() as session:
        result = session.run('MATCH (a:Article) WHERE a.abstract CONTAINS \'{test}\' RETURN a'.format(test=query))
    articles = [article['a'] for article in result.records()]
    # context = dict(query=result, papers=[dict(url='URL', relevancy=0.0, title='An example', abstract='Lorem ipsum imet ')])
    context = dict(papers=articles)
    return render(request, 'bird_results.html', context)
