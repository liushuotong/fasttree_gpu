#!/usr/bin/env python3
"""Create a source-only distribution from an explicit public manifest."""
import argparse
import gzip
import io
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def paths():
    names = [s.strip() for s in (ROOT/'release-files.txt').read_text().splitlines()
             if s.strip() and not s.lstrip().startswith('#')]
    if len(set(names)) != len(names):
        raise ValueError('duplicate public manifest path')
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or str(p) != name:
            raise ValueError(f'unsafe manifest path: {name}')
        source = ROOT/name
        if not source.is_file() or any(x.is_symlink() for x in [source, *source.parents]):
            raise ValueError(f'missing file or symlink in manifest: {name}')
    return sorted(names)


def ignore_text(names):
    return ('# Generated from release-files.txt by scripts/package_release.py --update-ignore\n'
            '# Keep local data, builds, credentials and historical evidence out of Git.\n'
            '*\n!*/\n' + ''.join('!/'+name+'\n' for name in names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--update-ignore', action='store_true')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    names = paths()
    expected = ignore_text(names)
    if args.update_ignore:
        (ROOT/'.gitignore').write_text(expected)
    if (ROOT/'.gitignore').read_text() != expected:
        raise SystemExit('stale .gitignore: run --update-ignore after editing the manifest')
    if args.check or args.update_ignore:
        print(f'Public manifest validated: {len(names)} files')
        return
    out = ROOT/'dist/fasttree_gpu-source.tar.gz'
    out.parent.mkdir(exist_ok=True)
    with out.open('wb') as raw, gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w') as archive:
            for name in names:
                data = (ROOT/name).read_bytes()
                info = tarfile.TarInfo('fasttree_gpu/'+name)
                info.size = len(data)
                info.mode = 0o755 if name.endswith(('.py','.sh')) else 0o644
                info.mtime = 0
                archive.addfile(info,io.BytesIO(data))
    print(f'{out}: {len(names)} source files')


if __name__ == '__main__':
    main()
