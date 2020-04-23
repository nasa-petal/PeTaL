from django.shortcuts import render
from django.http import HttpResponse

def form(request):
    context = dict(title='Form')
    return render(request, 'form.html', context)

def entry(request):
    print(request, flush=True)
    context = dict(title='Entry')
    return render(request, 'form.html', context)
