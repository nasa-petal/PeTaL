from django.shortcuts import render
from django.http import HttpResponse

from neo4j import GraphDatabase, basic_auth

def index(request):
    context = dict(query='')
    return render(request, 'bird/bird_e2b.html', context)

def results(request):
    query = request.GET.get('q')
    neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"))
    with neo_client.session() as session:
        result = session.run('MATCH (a:Article) WHERE a.abstract CONTAINS \'{test}\' RETURN a'.format(test=query))
    result = str(list(result.records())[0]['a']['abstract'])
    context = dict(query=result)
    return render(request, 'bird/bird_e2b.html', context)
