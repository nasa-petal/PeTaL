from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    context = dict(title='Home')
    return render(request, 'index.html', context)
