"""Stage 5: semantic retrieval + graph traversal.
Usage: python retrieve_legal.py "Someone hacked my account and stole money" --k 5
"""
import argparse
import json
import sys

from build_embeddings import embed_texts, load_model
from common import INDEX_NAME
from neo4j_http import Neo4jError, Neo4jQueryAPI

Q_SEARCH = """
CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node AS c, score
MATCH (p:LegalProvision)-[:HAS_CHUNK]->(c)
RETURN c.chunk_id AS chunk_id, c.text AS text, p.provision_id AS provision_id,
       p.section_number AS section_number, p.title AS title, p.act_name AS act,
       p.source_url AS source_url, c.page_start AS page_start, c.page_end AS page_end,
       p.penalty_text AS penalty_text, score
ORDER BY score DESC
"""

Q_RELATED = """
MATCH (p:LegalProvision) WHERE p.provision_id IN $ids
MATCH (p)-[r:RELATED_TO]-(q:LegalProvision)
RETURN p.provision_id AS for_provision, startNode(r).provision_id AS cites,
       q.provision_id AS provision_id, q.section_number AS section_number, q.title AS title,
       q.act_name AS act, q.source_url AS source_url, r.basis AS basis, r.evidence AS evidence
"""


def search(client, question, k=5, related=True, bundle=None):
    tok, model, device, max_len = bundle or load_model()
    emb = embed_texts([question], tok, model, device, max_len)[0]
    hits = client.run(Q_SEARCH, {"index": INDEX_NAME, "k": k, "embedding": emb}).rows
    rel = []
    if related and hits:
        rel = client.run(Q_RELATED, {"ids": sorted({h["provision_id"] for h in hits})}).rows
    return hits, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-related", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        hits, rel = search(Neo4jQueryAPI(), a.question, a.k, not a.no_related)
    except Neo4jError as e:
        print(f"Neo4j error: {e}")
        return 1
    if a.json:
        print(json.dumps({"results": hits, "related": rel}, indent=2, ensure_ascii=False))
        return 0
    if not hits:
        print("No results. Is the corpus ingested and the vector index ONLINE? (python verify_graph.py)")
        return 0
    for i, h in enumerate(hits, 1):
        print(f"\n#{i}  score={h['score']:.4f}  {h['act']} - Section {h['section_number']}: {h['title']}")
        print(f"    pages: {h['page_start']}-{h['page_end']} | source: {h['source_url']}")
        print(f"    text: {h['text'][:600]}{'...' if len(h['text']) > 600 else ''}")
        if h["penalty_text"]:
            print(f"    penalty (verbatim excerpt): {h['penalty_text'][:300]}")
    if rel:
        print("\nRelated provisions (explicit cross-references found in the source text):")
        for r in rel:
            arrow = "cites" if r["cites"] == r["for_provision"] else "is cited by"
            print(f"  {r['for_provision']} {arrow} -> {r['act']} s.{r['section_number']} {r['title']}")
    print("\nNote: Neo4j cosine scores are normalised to 0..1 ((1+cos)/2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
