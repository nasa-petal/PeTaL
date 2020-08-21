from django.urls import path
from django.views.generic import RedirectView
from .views import InputView, ResultView

urlpatterns = [
    # User input views
    path('',            RedirectView.as_view(url='bar/')),
    path('bar/',        InputView.as_view(template='bar.html'),       name='index'),
    path('dropdowns/',  InputView.as_view(template='dropdowns.html'), name='dropdowns'),
    path('nlp/',        InputView.as_view(template='query.html'),     name='nlp'),
    path('autocomplete/',InputView.as_view(template='autocomplete.html'), name='autocomplete'),

    # Article result views
    path('search_results/', ResultView.as_view(), name='bird_search'),
    ]
