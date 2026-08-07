#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from insplad_benchmark.dataset import exact_overlap_candidates, parse_supervised_manifest, parse_unsupervised_manifest


def jpeg_size(stream) -> tuple[int | None, int | None]:
    if stream.read(2) != b'\xff\xd8': return None, None
    while True:
        byte = stream.read(1)
        if not byte: return None, None
        if byte != b'\xff': continue
        marker = stream.read(1)
        while marker == b'\xff': marker = stream.read(1)
        if marker in {b'\xd8', b'\xd9'}: continue
        length = stream.read(2)
        if len(length) != 2: return None, None
        n = int.from_bytes(length, 'big')
        if n < 2: return None, None
        if marker and marker[0] in {0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF}:
            data = stream.read(n - 2)
            return (int.from_bytes(data[3:5], 'big'), int.from_bytes(data[1:3], 'big')) if len(data) >= 5 else (None, None)
        stream.seek(n - 2, io.SEEK_CUR)


def hash_member(zf: ZipFile, member: str) -> tuple[str, int | None, int | None]:
    h = hashlib.sha256()
    with zf.open(member) as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    w = hgt = None
    if member.lower().endswith(('.jpg', '.jpeg')):
        with zf.open(member) as f: w, hgt = jpeg_size(f)
    return h.hexdigest(), w, hgt


def main() -> int:
    import csv
    p = argparse.ArgumentParser(description='Confirm overlap candidates with SHA256 and JPEG dimensions without extracting ZIPs.')
    p.add_argument('--anomaly-zip', type=Path, required=True); p.add_argument('--fault-zip', type=Path, required=True)
    p.add_argument('--anomaly-manifest', type=Path, required=True); p.add_argument('--fault-manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('reports/image_overlap.csv'))
    p.add_argument('--summary', type=Path, default=Path('reports/image_overlap_summary.json'))
    args = p.parse_args()
    unsup = parse_unsupervised_manifest(args.anomaly_manifest); sup, _ = parse_supervised_manifest(args.fault_manifest)
    candidates = exact_overlap_candidates(sup, unsup); rows = []
    with ZipFile(args.anomaly_zip) as uz, ZipFile(args.fault_zip) as sz:
        uc, sc = {}, {}
        for i, row in enumerate(candidates, 1):
            up, sp = str(row['unsupervised_path']), str(row['supervised_path'])
            if up not in uc: uc[up] = hash_member(uz, up)
            if sp not in sc: sc[sp] = hash_member(sz, sp)
            usha, uw, uh = uc[up]; ssha, sw, sh = sc[sp]
            out = dict(row); out.update({'unsupervised_sha256': usha, 'supervised_sha256': ssha, 'sha256_match': usha == ssha, 'width': sw if sw == uw else None, 'height': sh if sh == uh else None, 'dimension_match': sw == uw and sh == uh, 'match_basis': 'sha256' if usha == ssha else 'manifest_candidate_only'}); rows.append(out)
            if i % 500 == 0: print(f'hashed {i}/{len(candidates)} candidates')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.out.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    by_asset = Counter(str(r['asset_category']) for r in rows if r['sha256_match'])
    summary = {
        'candidate_count': len(rows),
        'sha256_match_count': sum(bool(r['sha256_match']) for r in rows),
        'sha256_mismatch_count': sum(not bool(r['sha256_match']) for r in rows),
        'dimension_match_count': sum(bool(r['dimension_match']) for r in rows),
        'sha256_matches_by_asset': dict(sorted(by_asset.items())),
        'csv': str(args.out),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True); args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0

if __name__ == '__main__': raise SystemExit(main())
