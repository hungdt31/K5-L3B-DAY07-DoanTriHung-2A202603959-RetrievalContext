"""Offline tests for src/day7 (no API, no database). Fakes here only test code paths."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.day7.agent import parse_final
    from src.day7.retrieval import BM25Retriever, HybridRetriever, is_hit, matches, pack_context
except ImportError as exc:  # system python without the .venv dependencies
    raise unittest.SkipTest(f"run with .venv/bin/python ({exc})")


def chunk(i, text, page=1, inst="wbg"):
    return {"id": i, "source_id": "doc", "source": "doc.pdf", "page": page, "text": text,
            "metadata": {"institution": [inst]}}


CHUNKS = [chunk("a", "MIGA issued a record $10.6 billion in new guarantees", 38, "miga"),
          chunk("b", "IFC achieved record investment commitments of $105.6 billion", 32, "ifc"),
          chunk("c", "Trust funds held $27.9 billion in trust", 25)]


class FakeRanked:
    def __init__(self, order):
        self.order = order

    def search(self, query, top_k, filters=None):
        by = {c["id"]: c for c in CHUNKS}
        return [{**by[i], "score": 1.0, "rank": r} for r, i in enumerate(self.order[:top_k], 1)]


class Day7Tests(unittest.TestCase):
    def test_bm25_ranks_lexical_match_first_and_filters(self):
        bm25 = BM25Retriever(CHUNKS)
        self.assertEqual(bm25.search("record guarantees MIGA", 3)[0]["id"], "a")
        self.assertEqual([r["id"] for r in bm25.search("record", 3, {"institution": "ifc"})], ["b"])

    def test_bm25_rejects_unknown_filter_key(self):
        with self.assertRaises(ValueError):
            BM25Retriever(CHUNKS).search("x", 3, {"page": 1})

    def test_rrf_rewards_agreement(self):
        hybrid = HybridRetriever(FakeRanked(["b", "a", "c"]), FakeRanked(["a", "c", "b"]), candidates=3)
        results = hybrid.search("q", 3)
        self.assertEqual(results[0]["id"], "a")          # ranks 2 and 1 beat ranks 1 and 3
        self.assertEqual([r["rank"] for r in results], [1, 2, 3])

    def test_pack_context_dedup_and_budget(self):
        results = [{**CHUNKS[0]}, {**CHUNKS[0]}, {**CHUNKS[1]}]
        context, ids = pack_context(results, budget=90)
        self.assertEqual(ids, ["a"])
        self.assertIn("[a | doc | page 38]", context)

    def test_is_hit_requires_page_and_evidence(self):
        gold = [{"source_id": "doc", "page": 38, "evidence": "$10.6 billion"}]
        self.assertTrue(is_hit(CHUNKS[0], gold))
        self.assertFalse(is_hit({**CHUNKS[0], "page": 39}, gold))

    def test_matches_list_metadata(self):
        self.assertTrue(matches(CHUNKS[0], {"institution": "miga"}))
        self.assertFalse(matches(CHUNKS[0], {"institution": "ifc"}))

    def test_parse_final(self):
        ok = parse_final('```json\n{"status": "answered", "answer": "x [a]", "citations": ["a", 3]}\n```')
        self.assertEqual(ok["citations"], ["a"])
        self.assertIsNone(parse_final("0"))
        self.assertIsNone(parse_final('{"status": "maybe", "answer": "x"}'))


if __name__ == "__main__":
    unittest.main()
