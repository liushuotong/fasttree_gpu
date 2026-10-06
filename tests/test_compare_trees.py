"""Regression tests for evidence that distinguishes topology and all branch lengths."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'bench' / 'compare_trees.py'

class CompareTreesTest(unittest.TestCase):
    def run_comparison(self, a, b, *options):
        with tempfile.TemporaryDirectory() as tmp:
            f1, f2 = Path(tmp) / 'a.tree', Path(tmp) / 'b.tree'
            f1.write_text(a); f2.write_text(b)
            return subprocess.run([sys.executable, str(SCRIPT), str(f1), str(f2), *options], capture_output=True, text=True)

    def test_pendant_length_mismatch(self):
        self.assertEqual(self.run_comparison('(a:1,b:2,(c:3,d:4):5);', '(a:9,b:2,(c:3,d:4):5);').returncode, 1)

    def test_binary_root_edge_is_sum(self):
        self.assertEqual(self.run_comparison('(a:1,b:2,(c:3,d:4):5);', '((a:1,b:2):2,(c:3,d:4):3);').returncode, 0)

    def test_tolerance_option(self):
        a, b = '(a:1,b:2,c:3);', '(a:1.0001,b:2,c:3);'
        self.assertEqual(self.run_comparison(a, b, '--tolerance', '0.001').returncode, 0)
        self.assertEqual(self.run_comparison(a, b, '--tolerance', '0').returncode, 1)

    def test_nonfinite_and_duplicate_and_truncated(self):
        for bad in ['(a:nan,b:2,c:3);', '(a:1,a:2,c:3);', '(a:1,b:2', '(a:1,b:2,c:3);junk']:
            with self.subTest(bad=bad):
                self.assertEqual(self.run_comparison(bad, '(a:1,b:2,c:3);').returncode, 2)

    def test_topology_mismatch(self):
        self.assertEqual(self.run_comparison('(a:1,b:1,(c:1,d:1):1);', '(a:1,c:1,(b:1,d:1):1);').returncode, 1)

if __name__ == '__main__': unittest.main()
