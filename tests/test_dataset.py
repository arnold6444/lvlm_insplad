import csv
from pathlib import Path
from insplad_benchmark.dataset import build_anomaly_qa,build_defect_qa,category_intersection,normalize_defect,parse_supervised_manifest,parse_unsupervised_manifest,select_reference_by_asset

def write_manifest(path:Path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['path','is_dir','size','compressed_size','extension','crc32']); w.writeheader(); w.writerows(rows)

def test_normalization_and_structure(tmp_path):
    u=tmp_path/'u.csv'; s=tmp_path/'s.csv'
    write_manifest(u,[{'path':'asset-a/train/good/ref.jpg','is_dir':False,'size':10,'compressed_size':9,'extension':'.jpg','crc32':'aaaa'},{'path':'asset-a/test/good/a.jpg','is_dir':False,'size':11,'compressed_size':9,'extension':'.jpg','crc32':'bbbb'},{'path':'asset-a/test/rust/b.jpg','is_dir':False,'size':12,'compressed_size':9,'extension':'.jpg','crc32':'cccc'},{'path':'asset-b/test/good/c.jpg','is_dir':False,'size':13,'compressed_size':9,'extension':'.jpg','crc32':'dddd'}])
    write_manifest(s,[{'path':'defect_supervised/asset-a/train/normal/n.jpg','is_dir':False,'size':20,'compressed_size':9,'extension':'.jpg','crc32':'eeee'},{'path':'defect_supervised/asset-a/train/corrosão/x.jpg','is_dir':False,'size':21,'compressed_size':9,'extension':'.jpg','crc32':'ffff'},{'path':'defect_supervised/asset-a/val/good/v.jpg','is_dir':False,'size':22,'compressed_size':9,'extension':'.jpg','crc32':'1111'}])
    ur=parse_unsupervised_manifest(u); sr,_=parse_supervised_manifest(s); assert category_intersection(ur,sr)==['asset-a']; assert normalize_defect('corrosão')=='corrosao'; assert {r['defect_class'] for r in sr}=={'normal','corrosao'}
    qa,split,reason=build_defect_qa(sr); assert split=='val' and 'no test' in reason; assert qa[0]['answer']=={'anomaly':'normal','defect_class':'normal'}

def test_reference_excludes_test_duplicate(tmp_path):
    u=tmp_path/'u.csv'; write_manifest(u,[{'path':'asset-a/train/good/leaky.jpg','is_dir':False,'size':10,'compressed_size':9,'extension':'.jpg','crc32':'aaaa'},{'path':'asset-a/train/good/safe.jpg','is_dir':False,'size':20,'compressed_size':9,'extension':'.jpg','crc32':'bbbb'},{'path':'asset-a/test/good/test.jpg','is_dir':False,'size':10,'compressed_size':9,'extension':'.jpg','crc32':'aaaa'}])
    rows=parse_unsupervised_manifest(u); refs=select_reference_by_asset(rows,['asset-a'],seed=42); assert refs['asset-a']['relative_path'].endswith('safe.jpg')
    z=build_anomaly_qa(rows,['asset-a']); o=build_anomaly_qa(rows,['asset-a'],references=refs); assert [(r['id'],r['image'],r['answer']) for r in z]==[(r['id'],r['image'],r['answer']) for r in o]
