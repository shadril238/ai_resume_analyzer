# **Starting a Django Project in Conda Environment**

## **1. Activate Your Conda Environment**
If you already have a Conda environment, activate it:
```bash
conda activate shadril238  # Replace with your actual environment name
```
If you haven’t created one, do:
```bash
conda create --name shadril238 python=3.10 -y
conda activate shadril238
```

---

## **2. Install Project Dependencies**
If your project has a `requirements.txt` file, install dependencies:
```bash
pip install -r requirements.txt
```
Otherwise, manually install Django and DRF:
```bash
pip install django djangorestframework
```

---

## **3. Set Up Environment Variables**
If your project uses **`.env` files**, install `python-dotenv`:
```bash
pip install python-dotenv
```
Then create a `.env` file if it doesn’t exist:
```bash
touch .env
```
Add necessary environment variables inside `.env`:
```
DJANGO_SECRET_KEY=your_secret_key
DEBUG=True
DATABASE_URL=your_database_url
```

---

## **4. Apply Migrations**
Run the following command to apply database migrations:
```bash
python manage.py migrate
```

---

## **5. Create a Superuser (Optional)**
If your project has **Django Admin**, create an admin user:
```bash
python manage.py createsuperuser
```
Follow the prompts to set up a username and password.

---

## **6. Run the Django Server**
Start the development server:
```bash
python manage.py runserver
```
By default, it runs at `http://127.0.0.1:8000/`.

---

## **7. Open the Project in Your Browser**
- API endpoints (if using DRF): `http://127.0.0.1:8000/api/`
- Django Admin: `http://127.0.0.1:8000/admin/`

---

## **8. (Optional) Debugging Issues**
- Check if the correct Conda environment is activated:
  ```bash
  conda info --envs
  ```
- If `manage.py` doesn’t run, check installed packages:
  ```bash
  pip list
  ```

Now your **Django Conda project is running successfully!** 🚀


---

## Gradio UI: AI Resume Analyzer

This project now includes a standalone Gradio app that ranks resumes from a local folder or uploaded files using sentence-transformers embeddings. Optional features include reranking with a free Hugging Face cross-encoder and prompt refinement via local Ollama (Llama3).

### 1) Install extras
```
pip install -r requirements.txt
# Optional for reranking (uncomment in requirements.txt or install directly):
# pip install transformers torch
```

### 2) Launch Gradio
```
python gradio_app.py
```
Open the URL shown in the terminal (typically http://127.0.0.1:7860).

### 3) Use the UI
- Modes:
  - Folder Mode: provide path to a directory containing PDF/DOCX resumes.
  - Upload Mode: upload one or more PDF/DOCX resumes directly.
- Job Description / Query: text describing the role or requirements.
- Top K: number of top matches to display.
- Min Score: filter out low-relevance results.
- Keywords: comma-separated terms to boost (each unique hit adds to score).
- Keyword Boost: weight per unique keyword found in a resume.
- Use HF Reranker: enable cross-encoder reranking (BAAI/bge-reranker-base).
- Refine via Ollama: if `ollama serve` is running with `llama3.2` pulled, query refinement runs locally.
- Export CSV: download current ranking as CSV.
- Preview: view metadata and excerpt for a specific rank.

Notes:
- The embedding model is `all-MiniLM-L6-v2` (free on Hugging Face), fast and CPU-friendly.
- Reranking uses `BAAI/bge-reranker-base` if installed; otherwise the app still works with embeddings only.
- PDF parsing uses PyMuPDF; DOCX parsing uses `docx2txt`.

---

## JSON API: Rank Folder (Django)

Endpoint: `POST /rank-folder/`

Body (JSON):
```
{
  "folder_path": "/absolute/path/to/resumes",
  "query": "Senior Python developer with Django",
  "top_k": 10,
  "keywords": "python,django,rest,aws",
  "min_score": 0.2,
  "keyword_boost": 0.05,
  "use_ollama": false,
  "use_reranker": false
}
```

Response:
```
{
  "indexed": 12,
  "results": [
    {"rank": 1, "name": "...", "path": "...", "score": 0.87, "base_score": 0.82, "keyword_hits": 3, "excerpt": "..."},
    ...
  ]
}
```
