from django.urls import path, include
from .views import home

from rest_framework.routers import DefaultRouter
from .views import ResumeViewSet, extract_resume_text

router = DefaultRouter()
router.register(r'resumes', ResumeViewSet)
urlpatterns = [
    path('', home, name='home'),
    path('', include(router.urls)),
]

urlpatterns += [
    path('resumes/<int:resume_id>/extract/', extract_resume_text, name='extract_resume_text'),
]
