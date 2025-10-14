import glob
import os

import faiss
import fitz  # PyMuPDF
import requests
from django.core.files.base import ContentFile
from django.http import HttpResponse, JsonResponse, FileResponse
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from sentence_transformers import SentenceTransformer
from ranker import ResumeRanker

from .models import Resume
from .serializers import ResumeSerializer

# Ollama API URL
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"  # Change to your downloaded model name


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


@api_view(['POST'])
def bulk_upload_from_folder(request):
    """
    Upload all resume PDFs from a selected folder.
    """
    folder_path = request.data.get("folder_path")  # Get folder path from request

    if not folder_path or not os.path.exists(folder_path):
        return JsonResponse({"error": "Invalid folder path"}, status=400)

    allowed_extensions = {".pdf"}  # Only allow PDFs
    uploaded_count = 0
    errors = []

    for file_path in glob.glob(os.path.join(folder_path, "*.pdf")):
        try:
            file_name = os.path.basename(file_path)
            ext = os.path.splitext(file_name)[1].lower()

            if ext not in allowed_extensions:
                errors.append(f"Skipping {file_name} (Invalid file type)")
                continue

            with open(file_path, "rb") as f:
                resume = Resume(name=os.path.splitext(file_name)[0], file=ContentFile(f.read(), name=file_name))
                resume.save()
                uploaded_count += 1

        except Exception as e:
            errors.append(f"Error processing {file_path}: {str(e)}")

    return JsonResponse({
        "message": f"{uploaded_count} resumes uploaded successfully",
        "errors": errors if errors else "No errors",
    }, status=201 if uploaded_count else 400)


# Load embedding model (pre-trained SentenceTransformer)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Small & fast model

# FAISS index for fast similarity search
embedding_dim = 384  # Depends on the model used
faiss_index = faiss.IndexFlatL2(embedding_dim)
resume_store = {}  # Dictionary to map resume IDs to file paths

# Shared ranker instance for JSON ranking API
api_ranker = ResumeRanker()


def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)


@api_view(['POST'])
def index_resumes(request):
    """
    Extract text from all resumes, convert to embeddings, and store in FAISS.
    """
    global faiss_index, resume_store

    resumes = Resume.objects.all()
    if not resumes:
        return JsonResponse({"message": "No resumes available for indexing"}, status=400)

    for resume in resumes:
        pdf_path = resume.file.path
        text = extract_text_from_pdf(pdf_path)

        # Convert text to embedding
        embedding = embedding_model.encode(text, convert_to_numpy=True).reshape(1, -1)

        # Store in FAISS and maintain mapping
        faiss_index.add(embedding)
        resume_store[len(resume_store)] = pdf_path  # ID-to-file mapping

    return JsonResponse({"message": f"Indexed {len(resume_store)} resumes"}, status=200)


@api_view(['POST'])
def find_candidate(request):
    """
    Given a prompt, find the best-matching candidate and return the resume PDF.
    """
    global faiss_index, resume_store

    prompt = request.data.get("prompt")
    if not prompt:
        return JsonResponse({"error": "Missing prompt"}, status=400)

    # Use Ollama to refine the query
    data = {"model": MODEL_NAME, "prompt": f"Find the best candidate for: {prompt}", "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=data)
        response_json = response.json()
        refined_prompt = response_json.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": f"Ollama request failed: {str(e)}"}, status=500)

    # Convert refined query to embedding
    query_embedding = embedding_model.encode(refined_prompt, convert_to_tensor=True).numpy()

    # Find the closest resume in FAISS
    _, indices = faiss_index.search(query_embedding.reshape(1, -1), 1)

    # Retrieve the best-matching resume
    if indices[0][0] in resume_store:
        pdf_path = resume_store[indices[0][0]]
        return FileResponse(open(pdf_path, "rb"), as_attachment=True, filename=os.path.basename(pdf_path))

    return JsonResponse({"message": "No matching candidate found"}, status=404)


# Resume API ViewSet (CRUD Operations)
class ResumeViewSet(viewsets.ModelViewSet):
    queryset = Resume.objects.all()
    serializer_class = ResumeSerializer


@api_view(['POST'])
def rank_folder(request):
    """Rank resumes from a given folder_path using the lighter ranker."""
    folder_path = request.data.get("folder_path")
    query = request.data.get("query") or request.data.get("prompt") or ""
    top_k = int(request.data.get("top_k", 5))
    keywords = request.data.get("keywords", "")
    min_score = float(request.data.get("min_score", 0.0))
    kw_boost = float(request.data.get("keyword_boost", 0.05))
    use_ollama = bool(request.data.get("use_ollama", False))
    use_reranker = bool(request.data.get("use_reranker", False))

    if not folder_path or not os.path.isdir(folder_path):
        return JsonResponse({"error": "Invalid folder_path"}, status=400)
    if not query.strip():
        return JsonResponse({"error": "Missing query"}, status=400)

    try:
        n = api_ranker.index_folder(folder_path)
        kw_list = [k.strip() for k in keywords.split(',') if k.strip()]
        api_ranker.use_reranker = use_reranker
        results = api_ranker.rank(
            query,
            top_k=top_k,
            keywords=kw_list,
            keyword_boost=kw_boost,
            min_score=min_score,
            use_ollama_refine=use_ollama,
        )
        return JsonResponse({"indexed": n, "results": results})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
