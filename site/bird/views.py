from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic.base import View
import json

from time import time

from petal_site.search import search, plot
from petal_site import qbird

'''
# TODO
Due to lack of time, 
search_results -> elif action == "searchDropdown": and elif action == 'searchNLP':
do NOT filter for duplicates. 

# TODO
All search() function call should eventually become an api call to seperate server.
'''

class InputView(View):
    template = "bar.html"

    def get(self, request):
        return render(request, self.template)

class ResultView(View):
    '''
    Recieves user input and search neo4j database for results. Returns results to respective html page to be rendered.
    '''
    def get(self, request):
        query = request.GET.get('q')
        query = query.lower()
        action = request.GET.get('action')

        if action == 'plot':
            context = plot(query)
            return render(request, 'plot_results.html', context)

        elif action == 'search':
            context = search(query)
            context['parent'] = 'bar.html'

        elif action == 'searchAutocomplete':
            context = search(query)
            context['parent'] = 'autocomplete.html'

        elif action == 'searchDropdown':
            bioterms = query.split(', ')
            ### TODO The following code should eventually become one api call with multiple search params.
            all_articles = []
            for term in bioterms:
                article = search(term)
                all_articles.extend(article.get('articles'))
            ### TODO END
            context = {}
            context['articles'] = all_articles
            context['parent'] = 'dropdowns.html'

        elif action == 'searchNLP':
            exact_matches, exact_synonym_matches, partial_matches = qbird.process_with_nlp(query)            
            ### TODO The following code should eventually become one api call with multiple search params.
            all_papers = []
            for a_match in exact_synonym_matches:
                papers = search(a_match)
                all_papers.extend(papers.get('articles'))
            ### TODO END
            context = {}
            context['articles'] = all_papers
            context['parent'] = 'query.html'
            
        if len(context['articles']) > 0:
            return render(request, 'results.html', context)
        else:
            return render(request, 'no_results.html', context)

    def post(self, request):
        query = request.POST.get('user_input')
        context = search(query)
        return HttpResponse(json.dumps(context), content_type='application/json')
