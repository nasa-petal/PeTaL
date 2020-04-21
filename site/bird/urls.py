from django.urls import path

from . import views

urlpatterns = [
    path('search/', views.search_results, name='search'),
    path('plot/', views.Plot.as_view(), name='plot'),
    path('', views.index, name='index'),
    ]
