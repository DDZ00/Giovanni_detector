#!/usr/bin/env python3
"""Download the Giovanni corpus bundle and unzip it.

Usage:
    python3 scripts/export_corpus.py [--url URL] [--out DIR]

Defaults to https://192.168.1.69/giovanni/export/corpus.zip and ./corpus_export/.
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import ssl
import zipfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://192.168.1.69/giovanni/export/corpus.zip")
    ap.add_argument("--out", default="corpus_export")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification (self-signed LAN cert)")
    args = ap.parse_args()

    ctx = ssl._create_unverified_context() if args.insecure else None
    print(f"Fetching {args.url} ...", file=sys.stderr)
    with urllib.request.urlopen(args.url, context=ctx) as r:
        data = r.read()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(out)
    print(f"Extracted {len(data):,} bytes into {out}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
