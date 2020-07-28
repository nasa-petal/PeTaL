from django.shortcuts import render
from django.http import HttpResponse
import json

from time import time

from petal_site.search import search, plot
from petal_site import qbird

def index(request):
    return render(request, 'bar.html')

def dropdowns(request):
    return render(request, 'dropdowns.html')

def autocomplete(request):
    return render(request, 'autocomplete.html')

def nlp(request):
    return render(request, 'query.html')

def api(request):
    query = request.POST.get("user_input")
    context = search(query)
    return HttpResponse(
        json.dumps(context),
        content_type="application/json"
    )

def search_results(request):
    query = request.GET.get('q')
    query = query.lower()
    action = request.GET.get('action')

    if action == 'plot':
        context = plot(query)
        return render(request, 'plot_results.html', context)

    elif action == 'search':
        context = search(query)
        context["parent"] = "bar.html"

        if len(context['articles']) > 0:
            return render(request, 'results.html', context)
        else:
            return render(request, 'no_results.html', context)

    elif action == "searchDropdown":
        context = search(query)
        context["parent"] = "dropdowns.html"
        if len(context['articles']) > 0:
            return render(request, 'results.html', context)
        else:
            return render(request, 'no_results.html', context)

    elif action == "searchAutocomplete":
        context = search(query)
        context["parent"] = "autocomplete.html"
        if len(context['articles']) > 0:
            return render(request, 'results.html', context)
        else:
            return render(request, 'no_results.html', context)

    elif action == 'searchNLP':
        nlp = qbird.load_nlp()
        terms = qbird.load_dict("static/js/NTRS_data.js")
        question = query

        exact_matches, exact_synonym_matches, partial_matches = qbird.get_eng_terms(question, terms, nlp)
        all_matches = exact_matches | exact_synonym_matches | partial_matches
        print ("all_matches",len(all_matches))
        print ("exact_matches", len(exact_matches))
        print ("partial_matches", len(partial_matches))
        print ("exact_synonym_matches", len(exact_synonym_matches))

        all_papers =  []
        for a_match in exact_synonym_matches:
            papers = search(a_match)
            all_papers.extend(papers.get("articles"))

        context = {}
        context["articles"] = all_papers
        context["parent"] = "query.html"

        if len(context['articles']) > 0:
            return render(request, 'results.html', context)
        else:
            return render(request, 'no_results.html', context)
