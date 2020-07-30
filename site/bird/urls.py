from django.urls import path

from . import views

urlpatterns = [
    path('search/', views.search_results, name='bird_search'),
    path('', views.index, name='index'),
    path('bar/', views.index, name='index'),
    path('dropdowns/', views.dropdowns, name='dropdowns'),
    ]
