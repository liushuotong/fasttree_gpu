#!/usr/bin/env python3
"""Convert the interleaved PHYLIP alignments shipped with the FastTree samples
to FASTA (used to build small real-data inputs for tests/benchmarks).

usage: phylip2fasta.py input.p [output.fa] [maxSequences]
"""
import sys


def parse_phylip(path):
    lines = open(path).read().split('\n')
    hdr = lines[0].split()
    nseq, npos = int(hdr[0]), int(hdr[1])
    seqs = []
    cur = -1
    first_block = True
    for ln in lines[1:]:
        if not ln.strip():
            first_block = False
            cur = -1
            continue
        body = ln[11:].replace(' ', '')
        if first_block:
            name = ln[:11].strip()
            if name:
                seqs.append([name, body])
                cur = len(seqs) - 1
                continue
        cur += 1
        seqs[cur][1] += body
    assert len(seqs) == nseq, (len(seqs), nseq)
    assert all(len(s) == npos for _, s in seqs), [len(s) for _, s in seqs][:5]
    return seqs


def main():
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else None
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    seqs = parse_phylip(src)
    if limit:
        seqs = seqs[:limit]
    out = open(dst, 'w') if dst else sys.stdout
    for i, (n, s) in enumerate(seqs):
        out.write(">seq%d_%s\n%s\n" % (i, n, s))
    if dst:
        out.close()


if __name__ == '__main__':
    main()
