from django.shortcuts import render
from django.http import HttpResponse

from neo4j import GraphDatabase, basic_auth

from time import time

from .search import search

# TODO move the client elsewhere? This is just for optimization/testing - Lucas
neo_client = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", "life"), encrypted=False)
session = neo_client.session()

from django.views.generic.base import TemplateView

import plotly.express as px
import plotly.offline as opy
import plotly.graph_objs as go

class Plot(TemplateView):
    template_name = 'graph_results.html'

    def get_context_data(self, **kwargs):
        context = super(Plot, self).get_context_data(**kwargs)

        df = px.data.gapminder()
        df_2007 = df.query("year==2007")

        fig = px.scatter(df_2007, x="gdpPercap", y="lifeExp", size="pop", color="continent", log_x=True, size_max=60, template='plotly_dark', title="Gapminder 2007")

        div = opy.plot(fig, auto_open=False, output_type='div')
        context['graph'] = div
        return context

def index(request):
    context = dict(query='')
    return render(request, 'bar.html', context)

def search_results(request):
    query = request.GET.get('q')
    action = request.GET.get('action')
    if action == 'search':
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
            return render(request, 'results.html', context)
        else:
            return render(request, 'no_results.html', context)
    else:
        return Plot.as_view()(request)

