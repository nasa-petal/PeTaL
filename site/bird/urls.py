from django.urls import path

from . import views

urlpatterns = [
    path('search/', views.search_results, name='bird_search'),
    path('', views.index, name='index'),
    ]
