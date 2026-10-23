"""Stage 2: chunk provisions to the model's token limit and embed with law-ai/InLegalBERT.

Masked mean pooling + L2 normalisation. Incremental: unchanged chunks are not re-embedded.
Output: data/processed/chunks_embedded.jsonl, embedding_meta.json
"""
import argparse
import json
import sys

import torch
from transformers import AutoModel, AutoTokenizer

from common import (CHUNKS_FILE, EMB_META_FILE, EXPECTED_DIM, MODEL_NAME, PROVISIONS_FILE,
                    read_jsonl, sha, write_jsonl)


def load_model(name: str = MODEL_NAME):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    if not tok.is_fast:
        raise RuntimeError("A fast tokenizer is required (offset mapping) for exact chunk text.")
    tok.model_max_length = int(1e9)  # silence long-text warning; we truncate explicitly
    model = AutoModel.from_pretrained(name).to(device).eval()
    max_len = min(512, model.config.max_position_embeddings)
    return tok, model, device, max_len


@torch.no_grad()
def embed_texts(texts, tok, model, device, max_len=512, batch_size=16):
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).type_as(hidden)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # masked mean pooling
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)      # L2 norm
        out.extend([[round(x, 6) for x in v] for v in pooled.cpu().tolist()])
    return out


def chunk_provision(prov, tok, max_len, overlap=64):
    """Token-window chunks with exact source text (via char offsets). A context header is
    prepended ONLY for embedding, never stored as legal text."""
    header = f"{prov['act_name']} - Section {prov['section_number']}: {prov['title'][:150]}. "
    header_len = len(tok(header, add_special_tokens=False)["input_ids"])
    budget = max_len - 2 - header_len - 8
    if budget < 64:
        header, budget = f"Section {prov['section_number']}. ", max_len - 2 - 12
    text = prov["full_text"]
    if not text:
        return []
    offsets = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    n, start, idx, chunks = len(offsets), 0, 0, []
    while start < n:
        end = min(start + budget, n)
        piece = text[offsets[start][0]:offsets[end - 1][1]]
        chunks.append({
            "chunk_id": f"{prov['provision_id']}::c{idx}", "provision_id": prov["provision_id"],
            "chunk_index": idx, "text": piece, "embed_text": header + piece, "token_count": end - start,
            "section_number": prov["section_number"], "act_name": prov["act_name"],
            "source_url": prov["source_url"], "page_start": prov.get("page_start"),
            "page_end": prov.get("page_end"), "is_dummy": bool(prov.get("is_dummy", False)),
        })
        if end == n:
            break
        start, idx = end - overlap, idx + 1
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-embed everything")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    provisions = read_jsonl(PROVISIONS_FILE)
    if not provisions:
        print("No data/processed/provisions.jsonl. Run ingest_legal_data.py first (needs official PDFs).")
        return 2

    print(f"Loading {MODEL_NAME} ...")
    tok, model, device, max_len = load_model()
    probe = embed_texts(["dimension probe"], tok, model, device, max_len)[0]
    dim = len(probe)
    assert dim == model.config.hidden_size, "probe dimension != hidden_size"
    print(f"Device: {device} | max tokens: {max_len} | CONFIRMED embedding dimension: {dim}")
    if dim != EXPECTED_DIM:
        print(f"WARNING: planned dimension was {EXPECTED_DIM}; the vector index will be created with {dim}.")

    chunks = [c for p in provisions for c in chunk_provision(p, tok, max_len)]
    for c in chunks:
        c["text_hash"] = sha(MODEL_NAME + c["embed_text"])
    existing = {} if a.force else {c["chunk_id"]: c for c in read_jsonl(CHUNKS_FILE)}
    todo = [c for c in chunks if not (existing.get(c["chunk_id"]) and existing[c["chunk_id"]]["text_hash"] == c["text_hash"]
                                      and len(existing[c["chunk_id"]]["embedding"]) == dim)]
    print(f"{len(chunks)} chunks total; {len(todo)} need embedding; {len(chunks) - len(todo)} reused.")
    for i in range(0, len(todo), 64):
        part = todo[i:i + 64]
        for c, v in zip(part, embed_texts([x["embed_text"] for x in part], tok, model, device, max_len, a.batch_size)):
            c["embedding"] = v
        print(f"  embedded {min(i + 64, len(todo))}/{len(todo)}")
    for c in chunks:
        if "embedding" not in c:
            c["embedding"] = existing[c["chunk_id"]]["embedding"]
        c["embedding_model"], c["embedding_dim"] = MODEL_NAME, dim
        c.pop("embed_text", None)
    write_jsonl(CHUNKS_FILE, chunks)
    EMB_META_FILE.write_text(json.dumps({"model": MODEL_NAME, "dimension": dim, "max_tokens": max_len}), encoding="utf-8")
    print(f"Wrote {len(chunks)} embedded chunks (dim={dim}) -> {CHUNKS_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
