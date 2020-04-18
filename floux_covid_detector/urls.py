"""floux_covid_detector URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
"""
from django.urls import path, include

urlpatterns = [
    path('api/', include('floux_covid_detector_api.urls')),
]
