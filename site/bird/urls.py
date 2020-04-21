from django.urls import path

from . import views

urlpatterns = [
    path('search/', views.search_results, name='search'),
    path('', views.index, name='index'),
    ]
