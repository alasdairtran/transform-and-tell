from django.urls import path

from tat import views

urlpatterns = [
    path('caption/', views.post_caption),
]
