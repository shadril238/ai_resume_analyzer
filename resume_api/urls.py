from django.urls import path, include
from .views import home, bulk_upload_from_folder, index_resumes, find_candidate

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
    path('bulk-upload-folder/', bulk_upload_from_folder, name='bulk-upload-folder'),
    path('index-resumes/', index_resumes, name='index-resumes'),
    path('find-candidate/', find_candidate, name='find-candidate'),

]
