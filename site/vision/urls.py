from django.urls import path

from . import views

urlpatterns = [
    path('classify/', views.classify, name='classify'),
    path('', views.index, name='index'),
    ]
