"""Shared paths, config and small helpers."""
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROVISIONS_FILE = PROC_DIR / "provisions.jsonl"
RELATIONS_FILE = PROC_DIR / "relations.jsonl"
CHUNKS_FILE = PROC_DIR / "chunks_embedded.jsonl"
EMB_META_FILE = PROC_DIR / "embedding_meta.json"
REPORT_FILE = PROC_DIR / "ingest_report.json"

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "law-ai/InLegalBERT")
INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "legal_chunk_embedding")
ACT_NAME_PROPERTY = os.getenv("ACT_NAME_PROPERTY", "name")
EXPECTED_DIM = int(os.getenv("EXPECTED_EMBEDDING_DIM", "768"))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path):
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
