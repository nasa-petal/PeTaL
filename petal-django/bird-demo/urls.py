from django.urls import path

from . import views

urlpatterns = [
    path('bird-results/', views.results, name='bird-results'),
    path('', views.index, name='index'),
]
