import os
import glob
import json
from typing import Dict, List, Tuple, Sequence

import numpy as np

try:
    import fitz  # PyMuPDF
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyMuPDF (fitz) is required to read PDFs") from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    raise RuntimeError("sentence-transformers is required for embeddings") from e

try:
    import requests
except Exception:
    requests = None

try:
    import docx2txt  # type: ignore
except Exception:
    docx2txt = None


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    return v / norm


class ResumeRanker:
    """
    Lightweight resume ranker based on sentence-transformers embeddings,
    with optional cross-encoder reranking if transformers is available.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        reranker_model_name: str = "BAAI/bge-reranker-base",
        use_reranker: bool = False,
        ollama_url: str = "http://localhost:11434/api/generate",
        ollama_model: str = "llama3.2",
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self.use_reranker = use_reranker
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

        self.embedder = SentenceTransformer(self.embedding_model_name)

        # Lazy init for reranker to avoid heavy import if not requested
        self._reranker = None

        # In-memory index
        self.docs: List[Dict] = []  # {path, name, text}
        self.doc_embeddings: np.ndarray | None = None
        self._last_results: List[Dict] = []

    def _ensure_reranker(self):  # pragma: no cover
        if not self.use_reranker:
            return
        if self._reranker is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name)
            model.eval()

            def score_fn(query: str, passages: List[str]) -> List[float]:
                with torch.no_grad():
                    pairs = [(query, p) for p in passages]
                    batch = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
                    logits = model(**batch).logits.view(-1).tolist()
                return logits

            self._reranker = score_fn
        except Exception:
            # If reranker fails to load, disable it gracefully
            self._reranker = None
            self.use_reranker = False

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text() for page in doc)

    @staticmethod
    def extract_text_from_docx(docx_path: str) -> str:
        if docx2txt is None:
            raise RuntimeError("DOCX support requires docx2txt. Install docx2txt to enable .docx parsing.")
        text = docx2txt.process(docx_path)
        return text or ""

    def extract_text(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self.extract_text_from_pdf(file_path)
        if ext == ".docx":
            return self.extract_text_from_docx(file_path)
        raise ValueError(f"Unsupported file type: {ext}")

    def index_folder(self, folder_path: str, patterns: Tuple[str, ...] = ("*.pdf", "*.docx")) -> int:
        if not os.path.isdir(folder_path):
            raise ValueError(f"Invalid folder path: {folder_path}")

        files: List[str] = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(folder_path, pat)))

        files = sorted(set(files))
        self.docs.clear()

        for f in files:
            try:
                text = self.extract_text(f)
                self.docs.append({
                    "path": f,
                    "name": os.path.basename(f),
                    "text": text,
                })
            except Exception:
                # Skip unreadable files
                continue

        # Compute embeddings and store normalized for cosine similarity
        if self.docs:
            texts = [d["text"] for d in self.docs]
            emb = self.embedder.encode(texts, convert_to_numpy=True, batch_size=16, show_progress_bar=False)
            self.doc_embeddings = _normalize(emb.astype(np.float32))
        else:
            self.doc_embeddings = None

        return len(self.docs)

    def index_files(self, file_paths: Sequence[str]) -> int:
        self.docs.clear()
        for f in file_paths:
            if not os.path.isfile(f):
                continue
            try:
                text = self.extract_text(f)
                self.docs.append({
                    "path": f,
                    "name": os.path.basename(f),
                    "text": text,
                })
            except Exception:
                continue

        if self.docs:
            texts = [d["text"] for d in self.docs]
            emb = self.embedder.encode(texts, convert_to_numpy=True, batch_size=16, show_progress_bar=False)
            self.doc_embeddings = _normalize(emb.astype(np.float32))
        else:
            self.doc_embeddings = None
        return len(self.docs)

    def _refine_query_with_ollama(self, prompt: str) -> str:
        if not requests:
            return prompt
        try:
            payload = {"model": self.ollama_model, "prompt": f"Refine this hiring query: {prompt}", "stream": False}
            r = requests.post(self.ollama_url, json=payload, timeout=8)
            r.raise_for_status()
            data = r.json()
            refined = data.get("response") or ""
            refined = refined.strip()
            return refined if refined else prompt
        except Exception:
            return prompt

    def rank(
        self,
        query: str,
        top_k: int = 5,
        use_ollama_refine: bool = False,
        keywords: Sequence[str] | None = None,
        keyword_boost: float = 0.05,
        min_score: float = 0.0,
    ) -> List[Dict]:
        if not self.docs or self.doc_embeddings is None:
            return []

        q = query.strip()
        if not q:
            return []

        if use_ollama_refine:
            q = self._refine_query_with_ollama(q)

        q_emb = self.embedder.encode([q], convert_to_numpy=True)
        q_emb = _normalize(q_emb.astype(np.float32))

        # cosine similarity via dot product after normalization
        sims = (self.doc_embeddings @ q_emb.T).reshape(-1)
        order = np.argsort(-sims)

        # keyword boosting
        kw = [k.strip().lower() for k in (keywords or []) if k and k.strip()]

        def kw_hits(text: str) -> int:
            if not kw:
                return 0
            t = text.lower()
            seen = set()
            for k in kw:
                if k and k in t:
                    seen.add(k)
            return len(seen)

        base_idxs = order.tolist()
        scored: List[Tuple[int, float, int, float]] = []  # (idx, base, hits, final)
        for j in base_idxs:
            hits = kw_hits(self.docs[j]["text"])
            final = float(sims[j]) + keyword_boost * float(hits)
            scored.append((j, float(sims[j]), hits, final))

        # sort by final score
        scored.sort(key=lambda x: x[3], reverse=True)
        top_k = int(max(1, min(top_k, len(scored))))
        picked = scored[:top_k]

        candidates = []
        for i, (j, base, hits, final) in enumerate(picked):
            item = {
                "rank": i + 1,
                "name": self.docs[j]["name"],
                "path": self.docs[j]["path"],
                "score": float(final),
                "base_score": float(base),
                "keyword_hits": int(hits),
                "excerpt": self.docs[j]["text"][:800].replace("\n", " ") if self.docs[j]["text"] else "",
            }
            candidates.append(item)

        if self.use_reranker:
            self._ensure_reranker()
            if self._reranker is not None:
                passages = [self.docs[j]["text"] for j in idxs]
                scores = self._reranker(q, passages)
                reranked = sorted(
                    zip(candidates, scores), key=lambda x: x[1], reverse=True
                )
                candidates = [
                    {**c, "rerank_score": float(s)} for c, s in reranked
                ]
                # Re-number rank after reranking
                for r, item in enumerate(candidates, 1):
                    item["rank"] = r

        # min score filtering
        if min_score > 0:
            candidates = [c for c in candidates if c["score"] >= float(min_score)]

        self._last_results = candidates
        return candidates

    def last_results(self) -> List[Dict]:
        return list(self._last_results)


def main_cli():  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Rank resumes from a folder")
    parser.add_argument("folder", help="Folder containing resume PDFs")
    parser.add_argument("query", help="Job description or search prompt")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--use_reranker", action="store_true")
    parser.add_argument("--use_ollama", action="store_true")
    args = parser.parse_args()

    ranker = ResumeRanker(use_reranker=args.use_reranker)
    n = ranker.index_folder(args.folder)
    print(f"Indexed {n} resumes from {args.folder}")
    res = ranker.rank(args.query, top_k=args.top_k, use_ollama_refine=args.use_ollama)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main_cli()
