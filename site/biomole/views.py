from django.shortcuts import render
from django.http import HttpResponse

from time import time

from petal_site.search import biomole_search

def index(request):
    return render(request, 'biomole.html', dict())

def search_results(request):
    query = request.GET.get('q')
    action = request.GET.get('action')
    context = biomole_search(query)
    print(context)
    return render(request, 'biomole_results.html', context)
