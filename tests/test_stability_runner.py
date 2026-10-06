"""Benchmark evidence must retain commands and fail closed on bad executions."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('runner', Path(__file__).resolve().parents[1]/'bench/run_stability.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class StabilityRunnerTest(unittest.TestCase):
    def execute(self, scenario):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'build').mkdir()
            (root/'build/fasttree_gpu').write_text('binary fixture')
            (root/'input.fa').write_text('>a\nA\n>b\nA\n')
            (root/'cases.txt').write_text('tiny|-nt|input.fa\n')
            argv = ['runner','--build','build','--out','results','--cases','cases.txt',
                    '--modes','cpu,cuda','--repeats','1','--cpu','8']
            def fake_run(command, **kwargs):
                mode = command[command.index('-backend')+1]
                text = b'(a:1,b:1);\n'
                if scenario == 'mismatch' and mode == 'cuda': text = b'(a:2,b:1);\n'
                kwargs['stdout'].write(text)
                return SimpleNamespace(returncode=3 if scenario=='failure' and mode=='cuda' else 0)
            old = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(runner.sys,'argv',argv), patch.object(runner.os,'sched_getaffinity',return_value={8},create=True), patch.object(runner.subprocess,'run',side_effect=fake_run):
                    if scenario == 'ok': runner.main()
                    else:
                        with self.assertRaises(SystemExit) as caught: runner.main()
                        self.assertNotEqual(caught.exception.code,0)
                cpu = json.loads((root/'results/tiny.0.cpu.command.json').read_text())
                cuda = json.loads((root/'results/tiny.0.cuda.command.json').read_text())
                self.assertEqual(cpu[cpu.index('-backend')+1],'cpu')
                self.assertEqual(cuda[cuda.index('-backend')+1],'cuda')
            finally:
                os.chdir(old)

    def test_commands_are_kept_per_mode(self): self.execute('ok')
    def test_failed_run_is_not_success(self): self.execute('failure')
    def test_output_mismatch_is_not_success(self): self.execute('mismatch')


if __name__ == '__main__': unittest.main()
