from django.urls import path

from . import views

urlpatterns = [
    path('bird-legacy-results/', views.results, name='bird-legacy-results'),
    path('', views.index, name='bird-legacy'),
]
