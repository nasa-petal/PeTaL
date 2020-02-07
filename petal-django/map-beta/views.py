from django.shortcuts import render
from django.http import HttpResponse

def map(request):
    context = dict(title='Map')
    return render(request, 'map.html', context)
