#!/usr/bin/env python3
"""Check FastTree-compatible file/stdin/output/model CLI against pristine source."""
import argparse
from pathlib import Path
import subprocess
import tempfile


def run(binary, args, stdin=None):
    result = subprocess.run([str(binary), *args], input=stdin,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    if result.returncode:
        raise RuntimeError(f'{binary.name}: {args}: {result.stderr.decode(errors="replace")}')
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True)
    parser.add_argument('--reference', required=True)
    args = parser.parse_args()
    binary, reference = Path(args.binary).resolve(), Path(args.reference).resolve()
    root = Path(__file__).resolve().parents[1]
    cases = [([], 'aa24.fa'), (['-lg'], 'aa24.fa'), (['-wag'], 'aa24.fa'),
             (['-nt'], 'nt24.fa'), (['-nt','-gtr'], 'nt24.fa'),
             (['-nt','-gtr','-gamma'], 'nt24.fa'), (['-nt','-nocat'], 'nt24.fa'),
             (['-nt','-noml','-nosupport'], 'nt24.fa'), (['-nt','-fastest'], 'nt24.fa'),
             (['-nt','-makematrix'], 'nt24.fa')]
    for flags, filename in cases:
        data = root/'tests/data'/filename
        opts = ['-seed','12345',*flags]
        expected = run(reference, [*opts, str(data)])
        assert run(binary, [*opts,str(data)]) == expected, flags
        assert run(binary, opts, data.read_bytes()) == expected, ('stdin',flags)
    with tempfile.TemporaryDirectory(prefix='fasttree cli ') as tmp:
        tmp = Path(tmp)
        data = root/'tests/data/nt24.fa'
        expected = run(reference, ['-seed','12345','-nt',str(data)])
        out = tmp/'output tree.nwk'
        assert run(binary, ['-seed','12345','-nt','-out',str(out),str(data)]) == b''
        assert out.read_bytes() == expected
        # Starting tree and logging must not change the inferred result vs upstream.
        for flags in [['-nt','-intree',str(out)], ['-nt','-log',str(tmp/'run.log')]]:
            options = ['-seed','12345',*flags,str(data)]
            assert run(binary, options) == run(reference, options), flags
        # Two sequential PHYLIP alignments exercise -n and the input stream.
        entries = []
        for line in data.read_text().splitlines():
            if line.startswith('>'): entries.append('')
            else: entries[-1] += line.strip()
        phylip = f'{len(entries)} {len(entries[0])}\n' + ''.join(
            f's{i:<9} {seq}\n' for i,seq in enumerate(entries))
        multi = (phylip+'\n'+phylip).encode()
        opts = ['-nt','-n','2','-seed','12345']
        assert run(binary, opts, multi) == run(reference, opts, multi)
    for flag in ('-help','-expert'):
        result = subprocess.run([str(binary),flag],capture_output=True,timeout=20)
        assert result.returncode == 0 and b'fasttree_gpu' in result.stderr, flag
    result = subprocess.run([str(binary),'-not-a-fasttree-option'],capture_output=True,timeout=20)
    assert result.returncode != 0
    print('PASS: FastTree CLI compatibility (models, stdin, files, -out, -intree, -log, -n, help, errors)')


if __name__ == '__main__':
    main()
