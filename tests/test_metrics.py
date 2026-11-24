# -*- coding: utf-8 -*-
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiaccent import baseline as B  # noqa: E402
from aiaccent import io as I  # noqa: E402
from aiaccent import metrics as M  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_counts(self):
        turns = [["a" * 10, "b" * 20], ["c" * 5]]
        r = M.measure(turns)
        self.assertEqual(r["n_turns"], 2)
        self.assertEqual(r["n_chunks"], 3)
        self.assertEqual(r["turn_share_1"], 0.5)

    def test_end_punct(self):
        self.assertTrue(M.ends_with_punct("好的。"))
        self.assertTrue(M.ends_with_punct("真的吗？"))
        self.assertTrue(M.ends_with_punct("行～"))
        self.assertFalse(M.ends_with_punct("就这样 没别的"))

    def test_cv_zero_when_equal(self):
        r = M.measure([["x" * 10, "y" * 10, "z" * 10]])
        self.assertAlmostEqual(r["within_turn_cv_median"], 0.0, places=6)
        self.assertAlmostEqual(r["chunk_len_cv"], 0.0, places=6)

    def test_cv_positive_when_uneven(self):
        r = M.measure([["x" * 2, "y" * 40]])
        self.assertGreater(r["within_turn_cv_median"], 0.5)

    def test_entropy_uniform_is_max(self):
        # 四种段数各出现一次 → 熵 = log2(4) = 2
        turns = [["a"], ["a", "b"], ["a", "b", "c"], ["a", "b", "c", "d"]]
        r = M.measure(turns)
        self.assertAlmostEqual(r["turn_chunks_entropy"], 2.0, places=6)

    def test_share_3(self):
        turns = [["a", "b", "c"]] * 3 + [["a"]]
        r = M.measure(turns)
        self.assertAlmostEqual(r["turn_share_3"], 0.75)

    def test_quantiles_small(self):
        self.assertEqual(M._quantiles([5]), [5, 5, 5])
        self.assertEqual(M._quantiles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), [2, 6, 10])


class TestBaseline(unittest.TestCase):
    def _base(self):
        return {"metrics": {"chunk_len_p50": {"p10": 7, "p50": 9, "p90": 16}}}

    def test_dev_sign_and_magnitude(self):
        rows = B.compare({"chunk_len_p50": 25}, self._base())
        r = rows[0]
        self.assertEqual(r["verdict"], "above")
        self.assertAlmostEqual(r["dev"], (25 - 9) / (16 - 9), places=2)
        self.assertEqual(r["pct"], 90.0)      # 无分位表时，超出 p90 记 90

    def test_dev_below(self):
        rows = B.compare({"chunk_len_p50": 2}, self._base())
        r = rows[0]
        self.assertEqual(r["verdict"], "below")
        self.assertAlmostEqual(r["dev"], (2 - 9) / (9 - 7), places=2)
        self.assertEqual(r["pct"], 10.0)

    def test_in_band_is_zero(self):
        rows = B.compare({"chunk_len_p50": 10}, self._base())
        r = rows[0]
        self.assertEqual(r["verdict"], "in")
        self.assertEqual(r["dev"], 0.0)

    def test_pct_from_quantile_grid(self):
        q = [float(i) for i in range(101)]     # p10=10, p50=50, p90=90
        b = {"metrics": {"chunk_len_p50": {"p10": 10, "p50": 50, "p90": 90, "q": q}}}
        r = B.compare({"chunk_len_p50": 25.5}, b)[0]
        self.assertAlmostEqual(r["pct"], 25.5, places=1)

    def test_experimental_hidden_by_default(self):
        b = {"metrics": {"chunk_len_p50": {"p10": 7, "p50": 9, "p90": 16},
                         "sentence_len_entropy": {"p10": 1, "p50": 2, "p90": 3}}}
        rep = {"chunk_len_p50": 9, "sentence_len_entropy": 2}
        names = [r["metric"] for r in B.compare(rep, b)]
        self.assertNotIn("sentence_len_entropy", names)
        names = [r["metric"] for r in B.compare(rep, b, experimental=True)]
        self.assertIn("sentence_len_entropy", names)


class TestHedge(unittest.TestCase):
    def test_counts_each_occurrence(self):
        self.assertEqual(M.count_hedges("不过这个可能吧"), 2)

    def test_no_prefix_double_count(self):
        # 词表里没有"建议你"，只数一次
        self.assertEqual(M.count_hedges("建议你去看看"), 1)

    def test_clean_text_is_zero(self):
        self.assertEqual(M.count_hedges("今天吃了碗面"), 0)

    def test_metric_in_report_and_direction(self):
        r = M.measure([["不过我觉得可能", "行"]])
        self.assertIn("hedge_per_kchar", r)
        self.assertIn("hedge_chunk_share", r)
        self.assertGreater(r["hedge_per_kchar"], 0)
        self.assertEqual(M.DIRECTION["hedge_per_kchar"], -1)
        self.assertIn("hedge_per_kchar", B.CORE)


class TestIO(unittest.TestCase):
    def test_jsonl_roundtrip(self):
        turns = [["a", "b"], ["c"]]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for t in turns:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            self.assertEqual(I.read_corpus(p), turns)

    def test_txt_format(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("a\nb\n\nc\n")
            self.assertEqual(I.read_corpus(p), [["a", "b"], ["c"]])

    def test_text_object(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps({"text": "第一段\n\n第二段"}, ensure_ascii=False) + "\n")
            self.assertEqual(I.read_corpus(p), [["第一段", "第二段"]])


if __name__ == "__main__":
    unittest.main()
