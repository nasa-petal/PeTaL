from django.urls import path

from . import views

urlpatterns = [
    path('search/', views.search_results, name='biomole_search'),
    path('', views.index, name='index'),
    ]
