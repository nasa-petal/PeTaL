from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    context = dict(query='')
    return render(request, 'bird/bird_e2b.html', context)

def results(request):
    query = request.GET.get('q')
    # GET RESULTS(query)
    context = dict(query=query)
    return render(request, 'bird/bird_e2b.html', context)
