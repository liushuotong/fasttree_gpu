import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('summary', Path(__file__).resolve().parents[1]/'bench/summarize_stability.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def row(batch, repeat, mode, seconds):
    return dict(dir=batch, case='nt', repeat=str(repeat), mode=mode, seconds=seconds)


class StabilitySummaryTest(unittest.TestCase):
    def test_pair_identity_includes_batch(self):
        rows = [row('first',0,'cpu',100),row('first',0,'cuda',90),
                row('second',0,'cpu',20),row('second',0,'cuda',25)]
        value = summary.analyze(rows)[('nt','cuda')]
        self.assertEqual(value['n'],2)
        self.assertEqual(value['diff_min'],-5)
        self.assertEqual(value['diff_max'],10)
        self.assertEqual(value['diff_median'],2.5)
        self.assertEqual(value['wins'],1)

    def test_gpu_modes_are_not_averaged(self):
        rows = [row('a',0,'cpu',10),row('a',0,'cuda',8),row('a',0,'auto',9)]
        values = summary.analyze(rows)
        self.assertEqual(values[('nt','cuda')]['diff_median'],2)
        self.assertEqual(values[('nt','auto')]['diff_median'],1)

    def test_missing_and_duplicate_pairs_fail(self):
        for rows in [[row('a',0,'cuda',1)],
                     [row('a',0,'cpu',2),row('a',0,'cpu',3),row('a',0,'cuda',1)]]:
            with self.assertRaises(ValueError): summary.analyze(rows)

    def test_failed_unverified_and_nonfinite_runs_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'timings.tsv'
            for seconds, exact in [('FAILED_1','-'),('1','False'),('1','None'),('nan','True')]:
                p.write_text('case\trepeat\tmode\tseconds\texact_vs_cpu\nnt\t0\tcpu\t'+seconds+'\t'+exact+'\n')
                with self.assertRaises(ValueError): summary.load([tmp])


if __name__ == '__main__': unittest.main()
