from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    context = dict(title='Home')
    return render(request, 'bird/bird_e2b.html', context)

def results(request):
    context = dict(title='Results')
    return render(request, 'bird/bird_e2b.html', context)
