from django.urls import path

from . import views

urlpatterns = [
    path('', views.outer, name='biomole'),
    path('inner', views.inner, name='inner'),
]
