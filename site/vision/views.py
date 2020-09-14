from django.shortcuts import render
from django.http import HttpResponse

from time import time


def index(request):
    return render(request, 'vision.html')

def classify(request):
    # code
    return render(request, 'vision.html', response)

