from django.urls import path, include
from rest_framework import routers

from floux_covid_detector_api import views

router = routers.DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
    path('process-frame/', views.ProcessFrameView.as_view()),
]
