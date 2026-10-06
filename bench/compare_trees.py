#!/usr/bin/env python3
"""Unrooted split comparison for Newick trees (topology + branch lengths).

FastTree's Newick output order is not canonical, so comparing strings is not a
valid tree comparison.  This tool parses the trees, computes the set of
bipartitions (splits) induced by the edges, and reports

  * |symmetric difference| of the non-trivial split sets (Robinson-Foulds)
  * the largest branch-length difference among matching edges, including tips
  * the number of leaves and whether the leaf sets are identical

usage: compare_trees.py tree1 tree2 [--tolerance 1e-6] [--quiet]
exit status 0 requires identical leaf sets, topology and branch lengths within tolerance.
"""
import sys
import argparse
import math


class Node:
    __slots__ = ('name', 'length', 'children')

    def __init__(self, name=None, length=0.0, children=None):
        self.name = name
        self.length = length
        self.children = children or []


def tokenize(s):
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c in '(),;':
            yield c
            i += 1
        elif c.isspace():
            i += 1
        else:
            j = i
            while j < n and s[j] not in '(),;' and not s[j].isspace():
                j += 1
            yield s[i:j]
            i = j


def parse_newick(text):
    toks = list(tokenize(text))
    pos = 0

    def parse_node():
        nonlocal pos
        if pos >= len(toks): raise ValueError("truncated Newick")
        node = Node()
        if toks[pos] == '(':
            pos += 1
            while True:
                node.children.append(parse_node())
                if toks[pos] == ',':
                    pos += 1
                    continue
                if toks[pos] == ')':
                    pos += 1
                    break
                raise ValueError('malformed newick near %r' % (toks[pos],))
        # label and/or branch length
        if pos < len(toks) and toks[pos] not in '(),;':
            label = toks[pos]
            pos += 1
            if ':' in label:
                name, _, length = label.partition(':')
                node.name = name or None
                node.length = float(length or 0.0)
            else:
                node.name = label
                if pos < len(toks) and toks[pos] not in '(),;':
                    lab2 = toks[pos]
                    pos += 1
                    if ':' in lab2:
                        _, _, length = lab2.partition(':')
                        node.length = float(length or 0.0)
        return node

    root = parse_node()
    if pos < len(toks) and toks[pos] == ";": pos += 1
    if pos != len(toks): raise ValueError("trailing Newick content")
    return root


def collect(root, include_tips=False):
    """Returns (leaf_labels, splits) where splits maps frozenset(leafset)->length."""
    leaves = []

    def label(node):
        if not node.children:
            if node.name is None or node.name in leaves:
                raise ValueError("missing or duplicate leaf label")
            leaves.append(node.name)
            return frozenset([node.name])
        s = frozenset()
        for c in node.children:
            s |= label(c)
        return s

    def splits(node, all_leaves):
        """Returns (set_of_leaves_below, {split: length}) for edges below node."""
        if not node.children:
            return frozenset([node.name]), {}
        below = frozenset()
        out = {}
        for c in node.children:
            cb, cs = splits(c, all_leaves)
            for key, length in cs.items(): out[key] = out.get(key, 0.0) + length
            below |= cb
            # the edge above c carries length c.length
            other = all_leaves - cb
            if not math.isfinite(c.length): raise ValueError("non-finite branch length")
            if cb and other and (include_tips or (len(cb) > 1 and len(other) > 1)):
                key = cb if min(cb) <= min(other) else other
                # A degree-two root divides one unrooted edge into two parts.
                out[key] = out.get(key, 0.0) + c.length
        return below, out

    all_leaves = label(root)
    _, sp = splits(root, all_leaves)
    return set(leaves), sp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tree1')
    parser.add_argument('tree2')
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    tol, quiet = args.tolerance, args.quiet
    if not math.isfinite(tol) or tol < 0: parser.error('tolerance must be finite and nonnegative')
    try:
        with open(args.tree1) as f: t1 = parse_newick(f.read())
        with open(args.tree2) as f: t2 = parse_newick(f.read())
        l1, s1 = collect(t1)
        l2, s2 = collect(t2)
        _, all1 = collect(t1, include_tips=True)
        _, all2 = collect(t2, include_tips=True)
    except (OSError, ValueError, IndexError) as e:
        print('INVALID TREE: %s' % e, file=sys.stderr)
        return 2
    if l1 != l2:
        print('LEAF SETS DIFFER: %d vs %d leaves' % (len(l1), len(l2)))
        return 1
    only1 = set(s1) - set(s2)
    only2 = set(s2) - set(s1)
    common = set(s1) & set(s2)
    worst = 0.0
    worst_split = None
    for k in set(all1) & set(all2):
        d = abs(all1[k] - all2[k])
        if d > worst:
            worst = d
            worst_split = k
    if not quiet:
        print('leaves=%d splits: t1=%d t2=%d common=%d only1=%d only2=%d '
              'max_branch_diff=%.3g' % (len(l1), len(s1), len(s2), len(common), len(only1),
                                        len(only2), worst))
        if worst_split is not None and worst > tol:
            print('  worst split (%d taxa): |%.8g - %.8g|' %
                  (len(worst_split), all1[worst_split], all2[worst_split]))
    ok = (not only1) and (not only2) and worst <= tol
    if not quiet:
        print('RESULT: %s' % ('IDENTICAL TOPOLOGY' if ok else 'DIFFERENT'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
