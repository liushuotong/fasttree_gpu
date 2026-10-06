#!/usr/bin/env python3
"""Generate a simulated alignment with a known model, used to scale the
end-to-end benchmarks (the sample archives only contain 500-taxon alignments).

The simulator is deliberately simple and independent of FastTree:
  * random Yule tree with nTaxon leaves
  * Jukes-Cantor (or HKY-like) substitution along branches
  * optional gap process (independent deletion with probability pGap)

usage: simulate_alignment.py --taxa N --sites L [--seed S] [--gap 0.05]
                             [--model jc|hky] [--out FILE] [--tree FILE]
"""
import argparse
import random
import sys


def random_tree(n, rng):
    """Random Yule tree with n leaves (root = 0).

    Returns (parent, children, branchlength)."""
    children = {0: []}
    parent = {0: None}
    leaves = [0]
    next_node = 1
    while len(leaves) < n:
        i = rng.randrange(len(leaves))
        node = leaves.pop(i)
        a, b = next_node, next_node + 1
        next_node += 2
        children[node] = [a, b]
        children[a] = []
        children[b] = []
        parent[a] = node
        parent[b] = node
        leaves.append(a)
        leaves.append(b)
    blen = {}
    for node in parent:
        if parent[node] is not None:
            blen[node] = rng.expovariate(10.0)
    return parent, children, blen


def evolve(seq, blen, rng, model, kappa=2.0, freqs=None):
    if freqs is None:
        freqs = [0.25, 0.25, 0.25, 0.25]
    if blen <= 0:
        return list(seq)
    out = []
    for c in seq:
        # probability of at least one substitution
        pchange = 1.0 - pow(2.718281828459045, -4.0 / 3.0 * blen)
        if rng.random() < pchange:
            if model == 'jc':
                out.append(rng.randrange(4))
            else:
                # HKY-ish: transition (a<->g, c<->t) more likely
                if c in (0, 2):
                    pool = [0, 2, 1, 3]
                    w = [kappa / (kappa + 1), 1.0 / (kappa + 1) / 2, 1.0 / (kappa + 1) / 2]
                else:
                    pool = [1, 3, 0, 2]
                    w = [kappa / (kappa + 1), 1.0 / (kappa + 1) / 2, 1.0 / (kappa + 1) / 2]
                out.append(rng.choices(pool, weights=w)[0])
        else:
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--taxa', type=int, required=True)
    ap.add_argument('--sites', type=int, required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--gap', type=float, default=0.0)
    ap.add_argument('--model', default='jc', choices=['jc', 'hky'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--tree', default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n = args.taxa
    parent, children, blen = random_tree(n, rng)
    letters = "ACGT"

    # evolve sequences down the tree
    seqs = {}
    order = sorted(children.keys())
    root = [i for i in children if parent.get(i) is None]
    # find the root of the Yule tree: node 0 has no parent in this construction
    root = 0
    stack = [(root, [rng.randrange(4) for _ in range(args.sites)])]
    while stack:
        node, seq = stack.pop()
        if node < n:
            seqs[node] = seq
        for ch in children.get(node, []):
            stack.append((ch, evolve(seq, blen.get(ch, 0.1), rng, args.model)))
    if len(seqs) != n:
        sys.exit("internal error: only %d leaves" % len(seqs))

    with open(args.out, 'w') as f:
        for i in range(n):
            s = seqs[i]
            if args.gap > 0:
                s = ['-' if rng.random() < args.gap else letters[c] for c in s]
            else:
                s = [letters[c] for c in s]
            f.write('>sim%d\n%s\n' % (i, ''.join(s)))
    if args.tree:
        with open(args.tree, 'w') as f:
            f.write(newick(0, children, blen, n) + '\n')


def newick(node, children, blen, n, depth=0):
    if node < n:
        return 'sim%d:%.8f' % (node, blen.get(node, 0.0))
    subs = [newick(c, children, blen, n, depth + 1) for c in children.get(node, [])]
    return '(' + ','.join(subs) + ')%.8f' % blen.get(node, 0.0)


if __name__ == '__main__':
    main()
