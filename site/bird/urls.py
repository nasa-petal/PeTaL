from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('search_results/', views.search_results, name='bird_search'),
    path('bar/', views.index, name='index'),
    path('dropdowns/', views.dropdowns, name='dropdowns'),
    path('nlp/', views.nlp, name='index'),
    ]
