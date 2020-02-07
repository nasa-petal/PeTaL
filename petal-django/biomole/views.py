from django.shortcuts import render
from django.http import HttpResponse

def outer(request):
    context = dict()
    return render(request, 'biomole/BioMole-wrapper.html', context)

def inner(request):
    context = dict()
    return render(request, 'biomole/BioMole.html', context)
