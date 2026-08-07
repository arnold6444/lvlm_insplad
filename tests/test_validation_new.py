from insplad_benchmark.validation import validate_aligned_anomaly_sets,validate_qa_rows

def test_structured_gt_and_reference_validation():
    index={'a/test/x.jpg':{'asset_category':'a','split':'test','label':'anomalous','content_fingerprint':'1:x'},'a/train/ref.jpg':{'asset_category':'a','split':'train','label':'normal','content_fingerprint':'2:y'}}
    rows=[{'id':'1','task':'structured_explainability','image':'a/test/x.jpg','asset_category':'a','valid_defect_classes':['normal','rust'],'answer':{'anomaly':'anomalous','defect_class':'rust'}}]
    assert validate_qa_rows(rows,{'a'},{'normal','rust'},index).ok

def test_alignment_detects_mismatch():
    assert validate_aligned_anomaly_sets([{'id':'x','image':'a','answer':'normal'}],[{'id':'x','image':'b','answer':'normal'}])
