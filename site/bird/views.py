from django.shortcuts import render
from django.http import HttpResponse

from time import time

from .search import search, plot

def index(request):
    return render(request, 'bar.html', dict())

def search_results(request):
    query = request.GET.get('q')
    action = request.GET.get('action')
    if action == 'search':
        context = search(query)
        if len(articles) > 0:
            return render(request, 'results.html', context)
        else:
            return render(request, 'no_results.html', context)
    else:
        context = plot(query)
        return render(request, 'plot_results.html', context)

