import unittest
import os
import sys
import tempfile
import json

# Ensure src is in import path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ocr_segmentation import config_loader, pdf_to_images, bbox_sort

class TestOcrSegmentation(unittest.TestCase):
    def test_config_loader(self):
        # Test get_work
        mock_config = {
            "works": [
                {"id": "work-1", "name": "Work 1", "resources": []},
                {"id": "work-2", "name": "Work 2", "resources": []}
            ]
        }
        work = config_loader.get_work(mock_config, "work-2")
        self.assertEqual(work["name"], "Work 2")
        
        with self.assertRaises(ValueError):
            config_loader.get_work(mock_config, "non-existent")

    def test_parse_page_range(self):
        # Test full range
        self.assertEqual(pdf_to_images.parse_page_range(None, 10), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        # Test dash range
        self.assertEqual(pdf_to_images.parse_page_range("1-5", 10), [0, 1, 2, 3, 4])
        
        # Test comma-separated list
        self.assertEqual(pdf_to_images.parse_page_range("1,3,5", 10), [0, 2, 4])
        
        # Test mixed
        self.assertEqual(pdf_to_images.parse_page_range("1-3,5,7-8", 10), [0, 1, 2, 4, 6, 7])

    def test_bbox_sort(self):
        lines = [
            "Col 1 Top",     # Box 1 (Rightmost column, top)
            "Col 1 Bottom",  # Box 2 (Rightmost column, bottom)
            "Col 2 Top",     # Box 3 (Left column, top)
            "Col 2 Bottom"   # Box 4 (Left column, bottom)
        ]
        
        # Define coordinates:
        # Col 1: center x ~ 100
        # Col 2: center x ~ 50
        # Average width is ~20 pixels. Overlap threshold is ~16 pixels.
        # So x=100 and x=50 are far apart, they should cluster into 2 columns.
        bbox_meta = [
            {"x": 90, "y": 10, "w": 20, "h": 15},   # Col 1 Top
            {"x": 92, "y": 50, "w": 20, "h": 15},   # Col 1 Bottom
            {"x": 40, "y": 12, "w": 20, "h": 15},   # Col 2 Top
            {"x": 41, "y": 48, "w": 20, "h": 15}    # Col 2 Bottom
        ]
        
        # Shuffle inputs to verify sorting works regardless of input order
        shuffled_indices = [2, 0, 3, 1]
        shuffled_lines = [lines[i] for i in shuffled_indices]
        shuffled_bboxes = [bbox_meta[i] for i in shuffled_indices]
        
        sorted_lines = bbox_sort.sort(shuffled_lines, shuffled_bboxes)
        
        # Expected Han reading order: Column 1 (Right) -> Column 2 (Left)
        # Top-to-bottom within columns
        expected = [
            "Col 1 Top",
            "Col 1 Bottom",
            "Col 2 Top",
            "Col 2 Bottom"
        ]
        self.assertEqual(sorted_lines, expected)

if __name__ == "__main__":
    unittest.main()
