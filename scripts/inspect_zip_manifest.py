#!/usr/bin/env python3
"""List ZIP members without extracting images or assuming an InsPLAD folder schema."""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile
TEXT_EXTENSIONS={'.txt','.csv','.json','.jsonl','.xml','.yaml','.yml'}

def inspect_zip(zip_path: Path, out_csv: Path, out_json: Path, text_preview_json: Path|None=None, preview_bytes:int=65536)->None:
    previews=[]
    with ZipFile(zip_path) as zf:
        infos=zf.infolist(); rows=[]; extension_counts=Counter(); top_level_counts=Counter()
        for info in infos:
            p=Path(info.filename); suffix=p.suffix.lower()
            if not info.is_dir(): extension_counts[suffix or '<none>']+=1
            if p.parts: top_level_counts[p.parts[0]]+=1
            rows.append({'path':info.filename,'is_dir':info.is_dir(),'size':info.file_size,'compressed_size':info.compress_size,'extension':suffix,'crc32':f'{info.CRC:08x}'})
            if text_preview_json and not info.is_dir() and suffix in TEXT_EXTENSIONS:
                with zf.open(info) as f: data=f.read(preview_bytes)
                previews.append({'path':info.filename,'size':info.file_size,'truncated':info.file_size>preview_bytes,'preview':data.decode('utf-8',errors='replace')})
    out_csv.parent.mkdir(parents=True,exist_ok=True)
    with out_csv.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['path','is_dir','size','compressed_size','extension','crc32']); w.writeheader(); w.writerows(rows)
    summary={'zip':str(zip_path),'members':len(rows),'files':sum(not r['is_dir'] for r in rows),'directories':sum(r['is_dir'] for r in rows),'top_level_entries':dict(sorted(top_level_counts.items())),'extension_counts':dict(sorted(extension_counts.items())),'note':'This manifest intentionally does not infer asset categories, splits, labels, or defects.'}
    out_json.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    if text_preview_json:
        text_preview_json.parent.mkdir(parents=True,exist_ok=True); text_preview_json.write_text(json.dumps(previews,indent=2,ensure_ascii=False),encoding='utf-8')

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('zip_path',type=Path); p.add_argument('--out-csv',type=Path,required=True); p.add_argument('--out-json',type=Path,required=True); p.add_argument('--text-preview-json',type=Path); p.add_argument('--preview-bytes',type=int,default=65536); args=p.parse_args()
    try: inspect_zip(args.zip_path,args.out_csv,args.out_json,args.text_preview_json,args.preview_bytes)
    except (FileNotFoundError,BadZipFile) as exc: p.error(str(exc))
    return 0
if __name__=='__main__': raise SystemExit(main())
