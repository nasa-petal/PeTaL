from django.shortcuts import render
from django.http import HttpResponse

from time import time

def index(request):
    return render(request, 'biomole.html', dict())
