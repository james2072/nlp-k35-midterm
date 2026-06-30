import unittest
import os
import sys
import tempfile

# Ensure src is in import path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from sentence_alignment import aligner, char_aligner, corpus_builder

class TestSentenceAlignment(unittest.TestCase):
    def test_greedy_alignment(self):
        han = ["越南立國之始", "歷代君王相繼", "北屬千年之久"]
        viet = [
            "Thuở nước Việt Nam mới dựng nước.",
            "Các đời vua nối nhau trị vì.",
            "Nghìn năm bị phương Bắc đô hộ."
        ]
        
        # Test greedy alignment matching
        pairs = aligner.align(han, viet, method="greedy")
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs[0]["han"], "越南立國之始")
        self.assertEqual(pairs[0]["viet"], "Thuở nước Việt Nam mới dựng nước.")
        self.assertEqual(pairs[1]["han"], "歷代君王相繼")
        self.assertEqual(pairs[1]["viet"], "Các đời vua nối nhau trị vì.")

    def test_is_han_char(self):
        self.assertTrue(char_aligner.is_han_char("越"))
        self.assertTrue(char_aligner.is_han_char("南"))
        self.assertFalse(char_aligner.is_han_char("a"))
        self.assertFalse(char_aligner.is_han_char("."))
        self.assertFalse(char_aligner.is_han_char(" "))
        self.assertFalse(char_aligner.is_han_char("。"))

    def test_char_alignment_verification(self):
        # Mock dictionaries
        # S1 (SinoNom_Similar): visually similar
        # S2 (QuocNgu_SinoNom): readings
        mock_similar = {
            "榥": ["庚", "板", "釈"]
        }
        mock_quocngu = {
            "trăm": ["百", "釈"]
        }
        
        # Scenario:
        # han = "榥"
        # viet = "trăm"
        # Since '榥' is not in mock_quocngu['trăm'] (which is ['百', '釈']),
        # we look at mock_similar['榥'] = ['庚', '板', '釈'].
        # The intersection of S1 and S2 is {'釈'}.
        # Size of intersection is 1.
        # So '榥' is corrected to '釈'.
        res = char_aligner._verify_pair("榥", "trăm", mock_similar, mock_quocngu)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["char"], "釈")
        self.assertEqual(res[0]["status"], "corrected")

        # Test perfect match (sn in S2)
        mock_quocngu_perfect = {
            "nam": ["南"]
        }
        res_perfect = char_aligner._verify_pair("南", "nam", {}, mock_quocngu_perfect)
        self.assertEqual(res_perfect[0]["status"], "ok")
        self.assertEqual(res_perfect[0]["char"], "南")

        # Test mismatch (intersection size 0)
        res_error = char_aligner._verify_pair("南", "trăm", mock_similar, mock_quocngu)
        self.assertEqual(res_error[0]["status"], "error")

    def test_corpus_builder_build(self):
        pairs = [
            {"han": "南", "viet": "nam", "score": 0.9},
            {"han": "國", "viet": "quốc", "score": 0.8}
        ]
        rows = corpus_builder.build(pairs, "test-work", 3)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], "HVB_003.001.001.01")
        self.assertEqual(rows[0]["han_text"], "南")
        self.assertEqual(rows[1]["id"], "HVB_003.001.001.02")
        self.assertEqual(rows[1]["viet_text"], "quốc")

    def test_tsv_io(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_tsv = os.path.join(tmpdir, "corpus.tsv")
            rows = [
                {"id": "HVB_001.001.001.01", "han_text": "南", "viet_text": "nam", "align_score": 0.9},
                {"id": "HVB_001.001.001.02", "han_text": "國", "viet_text": "quốc", "align_score": 0.8}
            ]
            corpus_builder.write_tsv(rows, test_tsv)
            self.assertTrue(os.path.exists(test_tsv))

            # Test merge TSV
            test_tsv_2 = os.path.join(tmpdir, "corpus_2.tsv")
            rows_2 = [
                {"id": "HVB_002.001.001.01", "han_text": "山", "viet_text": "sơn", "align_score": 0.95}
            ]
            corpus_builder.write_tsv(rows_2, test_tsv_2)

            merged_tsv = os.path.join(tmpdir, "merged.tsv")
            corpus_builder.merge_tsv([test_tsv, test_tsv_2], merged_tsv)
            
            with open(merged_tsv, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # 1 header + 2 from first file + 1 from second file = 4 lines
                self.assertEqual(len(lines), 4)
                self.assertTrue(lines[0].startswith("id\than_text"))
                self.assertTrue("HVB_001" in lines[1])
                self.assertTrue("HVB_002" in lines[3])

if __name__ == "__main__":
    unittest.main()
