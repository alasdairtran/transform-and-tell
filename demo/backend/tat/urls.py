from django.urls import path

from tat import views

urlpatterns = [
    path('scrape/', views.get_image_urls),
    path('caption/', views.post_caption),
]
