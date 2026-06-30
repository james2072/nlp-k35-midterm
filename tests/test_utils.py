import unittest
import os
import tempfile
import sys

# Ensure src is in import path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import utils

class TestUtils(unittest.TestCase):
    def test_generate_id(self):
        self.assertEqual(utils.generate_id(1, 2, 3, 4), "HVB_001.002.003.04")
        self.assertEqual(utils.generate_id(999, 999, 999, 99), "HVB_999.999.999.99")
        self.assertEqual(utils.generate_id(0, 0, 0, 0), "HVB_000.000.000.00")

    def test_json_io(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.json")
            data = {"key": "value", "unicode_str": "日本語/Tiếng Việt"}
            
            utils.write_json(data, test_file)
            self.assertTrue(os.path.exists(test_file))
            
            loaded = utils.read_json(test_file)
            self.assertEqual(data, loaded)

    def test_lines_io(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.txt")
            lines = ["  line 1  ", "", "line 2", "  "]
            
            utils.write_lines(lines, test_file)
            self.assertTrue(os.path.exists(test_file))
            
            # read_lines should strip lines and skip empty ones
            loaded = utils.read_lines(test_file)
            self.assertEqual(loaded, ["line 1", "line 2"])

if __name__ == "__main__":
    unittest.main()
