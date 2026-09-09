#!/usr/bin/env python3
"""Download hash-pinned benchmark files; verify every file before atomic install.

python tools/research/fetch_artifacts.py benchmarks/manifests/kws.json /path/cache
Uses raw GitHub at an immutable source revision. Existing verified files are reused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import os
import tempfile


def fetch(manifest,destination):
    revision=manifest['revision']
    if len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('expected immutable Git commit')
    for entry in manifest['files']+[manifest['evaluator_rules']]:
        rel=Path(entry['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe artifact path')
        target=destination/rel
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==entry['sha256']:
            continue
        url=f'https://raw.githubusercontent.com/mlcommons/tiny/{revision}/{rel.as_posix()}'
        with urllib.request.urlopen(url,timeout=60) as response:data=response.read()
        if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError(f'hash mismatch: {rel}')
        target.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(dir=target.parent,prefix=target.name+'.')
        try:
            with os.fdopen(fd,'wb') as f:f.write(data)
            os.replace(tmp,target)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('manifest',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args();fetch(json.loads(args.manifest.read_text()),args.destination)
