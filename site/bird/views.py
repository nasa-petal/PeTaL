from django.shortcuts import render
from django.http import HttpResponse

from time import time

from petal_site.search import search, plot

def index(request):
    return render(request, 'bar.html', dict())

def search_results(request):
    query = request.GET.get('q')
    action = request.GET.get('action')
    if action == 'plot':
        context = plot(query)
        return render(request, 'plot_results.html', context)
    else:
        context = search(query)
        if len(context['articles']) > 0:
            return render(request, 'results.html', context)
        else:
            return render(request, 'no_results.html', context)
