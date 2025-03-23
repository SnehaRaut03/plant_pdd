from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('get-treatment/<str:disease_name>/', views.get_treatment, name='get_treatment'),
     path('get-requirements/<str:plant_name>/', views.get_requirements, name='get_requirements'),
]
