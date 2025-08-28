from __future__ import annotations
import io, re
from typing import Tuple
import pdfplumber
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import textstat

EMBED = None

STRONG_LOR_PHRASES = [
    "strongly recommend", "highest recommendation", "without reservation",
    "top percentile", "outstanding", "exceptional", "exemplary",
]

def _init_model():
    global EMBED
    if EMBED is None:
        EMBED = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    # Prefer pdfplumber; fallback to pypdf
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

def _clean(txt: str) -> str:
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def _coherence_score(txt: str) -> float:
    _init_model()
    # Split into sentences naively
    sents = re.split(r"(?<=[.!?])\s+", txt)
    sents = [s.strip() for s in sents if len(s.strip()) > 0]
    if len(sents) < 3:
        return 0.4  # too short to judge
    embs = EMBED.encode(sents, normalize_embeddings=True)
    sims = [float(util.cos_sim(embs[i], embs[i+1])) for i in range(len(embs)-1)]
    # coherence favors consistent flow without being repetitive
    avg = sum(sims)/len(sims)
    return max(0.0, min(1.0, (avg + 1)/2))  # map [-1,1] → [0,1]

def _length_score(words: int, kind: str) -> float:
    # SOP ideal 700–1200 words; LOR 400–900 words
    lo, hi = (700, 1200) if kind == "sop" else (400, 900)
    if words <= 50:
        return 0.0
    if words < lo:
        return max(0.0, (words-50)/(lo-50)) * 0.7
    if words > hi:
        return max(0.3, 1 - (words-hi)/(hi))
    return 1.0

def _readability_score(txt: str) -> float:
    try:
        fre = textstat.flesch_reading_ease(txt)
    except Exception:
        return 0.5
    # Target moderate academic readability ~30–70
    if fre <= 0:
        return 0.2
    if fre >= 90:
        return 0.4
    # map [10, 90] → [0.6, 1.0], with best around 50
    diff = abs(fre-50)
    return max(0.0, 1 - diff/80)

def _lex_diversity(txt: str) -> float:
    words = re.findall(r"[A-Za-z']+", txt.lower())
    if len(words) < 50:
        return 0.4
    uniq = len(set(words))
    return min(1.0, uniq/len(words) * 2)  # cap at 1.0

def _phrase_boost(txt: str) -> float:
    t = txt.lower()
    hits = sum(1 for p in STRONG_LOR_PHRASES if p in t)
    return min(0.1 * hits, 0.3)  # at most +0.3

def score_document(file_bytes: bytes, filename: str, kind_hint: str | None = None) -> Tuple[float, dict]:
    """Return (score_0_1, details). kind_hint in {"lor","sop",None}.
    Heuristic scoring based on length, coherence, readability, lexical diversity, and strong phrases (for LOR).
    """
    raw = extract_text(file_bytes, filename)
    txt = _clean(raw)
    words = len(re.findall(r"[A-Za-z']+", txt))
    kind = kind_hint or ("lor" if any(w in txt.lower() for w in ["recommend", "reference"]) else "sop")

    sc_len = _length_score(words, kind)
    sc_coh = _coherence_score(txt)
    sc_read = _readability_score(txt)
    sc_lex = _lex_diversity(txt)
    boost = _phrase_boost(txt) if kind == "lor" else 0.0

    # weights sum to 1, plus small boost
    score = 0.30*sc_len + 0.35*sc_coh + 0.20*sc_read + 0.15*sc_lex
    score = min(1.0, max(0.0, score + boost))

    details = {
        "kind": kind,
        "words": words,
        "len": round(sc_len,3),
        "coherence": round(sc_coh,3),
        "readability": round(sc_read,3),
        "lexical_diversity": round(sc_lex,3),
        "phrase_boost": round(boost,3),
    }
    return float(score), details
