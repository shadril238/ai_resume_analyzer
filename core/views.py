from django.http import HttpResponse
from rest_framework import viewsets
from .models import Resume
from .serializers import ResumeSerializer
import fitz  # PyMuPDF
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Resume


# Basic API health check response
def home(request):
    return HttpResponse("AI Resume Analyzer is up and running!")

@api_view(['GET'])
def extract_resume_text(request, resume_id):
    try:
        resume = Resume.objects.get(id=resume_id)
        file_path = resume.file.path  # Get resume file path

        with fitz.open(file_path) as doc:
            text = "\n".join(page.get_text() for page in doc)  # Extract text

        return Response({"resume_text": text})
    except Resume.DoesNotExist:
        return Response({"error": "Resume not found"}, status=404)

# Resume API ViewSet (CRUD Operations)
class ResumeViewSet(viewsets.ModelViewSet):
    queryset = Resume.objects.all()
    serializer_class = ResumeSerializer

    #git branching test
