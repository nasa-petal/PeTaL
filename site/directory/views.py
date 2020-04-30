from django.shortcuts import render
from django.http import HttpResponse

from neo4j import GraphDatabase, basic_auth
neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"), encrypted=False)
session = neo_client.session()

def form(request):
    context = dict(title='Form')
    return render(request, 'form.html', context)

def entry(request):
    form = request.POST
    print(form, flush=True)
    name         = form['name']
    organization = form['organization']
    application  = form['application']
    tools        = form['tools']
    session.run('MERGE (a:Application {name: \'' + application + '\'})')
    for tool in tools.split(','):
        session.run('MERGE (t:Tool {name: \'' + tool + '\'})')
        session.run('MATCH (a:Application) WHERE a.name = \'' + application + '\' MATCH (t:Tool) WHERE t.name = \'' + tool + '\'' + 'MERGE (a)-[:uses]->(t)')
    session.run('MERGE (u:User {name: \'' + name + '\', organization: \'' + organization + '\'})')
    session.run('MATCH (u:User) WHERE u.name = \'' + name + '\' MATCH (a:Application) WHERE a.name = \'' + application + '\' MERGE (u)-[:works_on]->(a)')
    return render(request, 'done.html', dict())
