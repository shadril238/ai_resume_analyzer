import os
import csv
import tempfile
from typing import List

import gradio as gr

from ranker import ResumeRanker


ranker = ResumeRanker()
LAST_CSV_PATH = None


def index_resumes(folder_path: str) -> str:
    try:
        n = ranker.index_folder(folder_path)
        return f"Indexed {n} resumes from: {folder_path}"
    except Exception as e:
        return f"Error: {e}"


def index_uploads(files: List[str]) -> str:
    try:
        file_paths = files or []
        n = ranker.index_files(file_paths)
        return f"Indexed {n} uploaded files"
    except Exception as e:
        return f"Error: {e}"


def analyze(job_description: str, top_k: int, use_reranker: bool, use_ollama: bool, keywords: str, min_score: float, kw_boost: float):
    # Update toggles globally
    ranker.use_reranker = use_reranker
    kw_list = [k.strip() for k in (keywords or '').split(',') if k.strip()]

    results = ranker.rank(
        job_description,
        top_k=top_k,
        use_ollama_refine=use_ollama,
        keywords=kw_list,
        keyword_boost=float(kw_boost),
        min_score=float(min_score),
    )

    if not results:
        return [["-", "", "", 0.0, 0.0, 0, "No results. Index a folder or upload files and provide a query."]]

    table_rows = []
    for r in results:
        table_rows.append([
            r.get("rank"),
            r.get("name"),
            r.get("path"),
            round(float(r.get("score", 0.0)), 4),
            round(float(r.get("base_score", 0.0)), 4),
            int(r.get("keyword_hits", 0)),
            r.get("excerpt"),
        ])
    return table_rows


def export_csv() -> str:
    global LAST_CSV_PATH
    rows = ranker.last_results()
    if not rows:
        return ""
    fd, path = tempfile.mkstemp(prefix="resume_ranking_", suffix=".csv")
    os.close(fd)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "name", "path", "score", "base_score", "keyword_hits", "excerpt"])
        writer.writeheader()
        writer.writerows(rows)
    LAST_CSV_PATH = path
    return path


def reset_index():
    ranker.docs.clear()
    ranker.doc_embeddings = None
    return "Index cleared"


def preview_rank(n: int):
    rows = ranker.last_results()
    if not rows:
        return "No results to preview"
    n = int(n)
    if n < 1 or n > len(rows):
        return f"Rank must be between 1 and {len(rows)}"
    r = rows[n - 1]
    meta = f"#{r['rank']} — {r['name']}\n{r['path']}\nscore={r['score']:.4f} (base={r['base_score']:.4f}, kw_hits={r['keyword_hits']})\n\n"
    return meta + (r.get("excerpt") or "")


with gr.Blocks(title="AI Resume Analyzer") as demo:
    gr.Markdown(
        """
        # AI Resume Analyzer
        - Index resumes from a folder, or upload files (PDF/DOCX)
        - Enter a job description or search query
        - Optionally boost by keywords and refine query with Ollama
        - Export results as CSV and preview any ranked resume
        """
    )

    with gr.Tabs():
        with gr.Tab("Folder Mode"):
            with gr.Row():
                folder_inp = gr.Textbox(label="Folder Path (PDF/DOCX)", placeholder="/path/to/resumes")
                idx_btn = gr.Button("Index Folder")
            status1 = gr.Textbox(label="Status", interactive=False)
            idx_btn.click(index_resumes, inputs=[folder_inp], outputs=[status1])

        with gr.Tab("Upload Mode"):
            up_files = gr.Files(label="Upload PDF or DOCX resumes", file_count="multiple", type="filepath")
            idx_btn2 = gr.Button("Index Uploads")
            status2 = gr.Textbox(label="Status", interactive=False)
            idx_btn2.click(index_uploads, inputs=[up_files], outputs=[status2])

    gr.Markdown("## Search and Rank")
    job_desc = gr.Textbox(label="Job Description / Query", lines=6, placeholder="e.g., Senior Python developer with Django and REST experience")

    with gr.Row():
        topk_inp = gr.Slider(1, 100, value=10, step=1, label="Top K")
        min_score = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Min Score (filter)")
        kw_boost = gr.Slider(0.0, 0.5, value=0.05, step=0.01, label="Keyword Boost per Hit")
    keywords = gr.Textbox(label="Keywords (comma-separated)", placeholder="python, django, rest, aws")

    with gr.Row():
        use_reranker = gr.Checkbox(label="Use HF Reranker (BAAI/bge-reranker-base)", value=False)
        use_ollama = gr.Checkbox(label="Refine query via Ollama (llama3.2)", value=False)

    run_btn = gr.Button("Analyze & Rank", variant="primary")

    result_tbl = gr.Dataframe(
        headers=["rank", "name", "path", "score", "base_score", "keyword_hits", "excerpt"],
        datatype=["number", "str", "str", "number", "number", "number", "str"],
        row_count=(0, "dynamic"),
        wrap=True,
        label="Ranked Resumes"
    )

    with gr.Row():
        csv_btn = gr.Button("Export CSV")
        csv_file = gr.File(label="Download CSV")
        reset_btn = gr.Button("Reset Index")
        reset_out = gr.Textbox(label="Reset Status", interactive=False)

    with gr.Row():
        preview_rank_n = gr.Number(value=1, precision=0, label="Preview Rank #")
        preview_btn = gr.Button("Preview")
    preview_box = gr.Textbox(label="Preview", lines=12)

    run_btn.click(
        analyze,
        inputs=[job_desc, topk_inp, use_reranker, use_ollama, keywords, min_score, kw_boost],
        outputs=[result_tbl]
    )
    csv_btn.click(export_csv, outputs=[csv_file])
    reset_btn.click(reset_index, outputs=[reset_out])
    preview_btn.click(preview_rank, inputs=[preview_rank_n], outputs=[preview_box])


if __name__ == "__main__":
    demo.launch()
