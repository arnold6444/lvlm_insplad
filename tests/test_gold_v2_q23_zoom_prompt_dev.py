from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import run_gold_v2_q23_zoom_prompt_dev as runner
import test_gold_v2_verify_agent as fixtures


def setup(tmp, n=6):
    rows = [r for r in fixtures._rows(n) if r["question_type"] != runner.agent.Q3]
    first, _ = fixtures._stage(rows)
    qa = tmp/"dev/dev_master_qa.jsonl"
    runner.common.save_records(qa, rows)
    runner.old.atomic_json(qa.parent/"build_summary.json", {"quality_status": "FROZEN", "file": qa.name})
    sup, unsup = tmp/"sup.zip", tmp/"unsup.zip"
    with ZipFile(sup, "w") as z:
        for r in rows[::2]:
            data = fixtures._png(30)
            if r["core_id"] == "c1":
                b=io.BytesIO(); Image.new("RGB", (1080,1080), "white").save(b,format="PNG"); data=b.getvalue()
            z.writestr(r["image"], data)
    refs = [{"image": r["image"], "retrieved_diffdate": r["asset_category"]+"/train/good/ref.png"} for r in rows[::2]]
    with ZipFile(unsup, "w") as z:
        for member in sorted({r["retrieved_diffdate"] for r in refs}):z.writestr(member, fixtures._png(200))
    refpath=tmp/"refs/retrieved_refs.csv"
    runner.agent.cons.write_csv(refpath, refs)
    baseline=tmp/"baseline/predictions"/runner.MODEL/"normal_ref_retrieved_diffdate.jsonl"
    runner.common.save_records(baseline, [{"id": r["id"], "core_id": r["core_id"], "model_key":runner.MODEL,
        "condition":"normal_ref_retrieved_diffdate", "prompt_version":1, "output":first[r["id"]],
        "predicted_choice":first[r["id"]], "error":"", "latency_s":.5} for r in rows])
    out=tmp/"gold_v2_q23_zoom_prompt_dev_test"
    args=["--dev-qa",str(qa),"--stage1-dir",str(baseline.parents[1]),"--refs-csv",str(refpath),
        "--sup-zip",str(sup),"--unsup-zip",str(unsup),"--out-dir",str(out),"--n-images",str(n),
        "--n-boot","10","--checkpoint-every","2"]
    return rows, first, args, out, baseline


def fake(calls, fail=None, invalid=None, interrupt=None):
    def infer(adapter, images, prompt):
        core=Path(images[-1]).stem
        texts=dict(re.findall(r"^([A-J])\. (.+)$",prompt,re.M))
        step=("q2" if "Visual definitions" in prompt else "presence" if "A visible localized defect region." in texts.values()
              else "position" if "Use the full TARGET frame" in prompt else "direct")
        info=(core,step,adapter["image_tile_budgets"],prompt)
        calls.append(info)
        if interrupt and interrupt(info):raise KeyboardInterrupt()
        if fail and fail(info):raise RuntimeError("fixture transient inference failure")
        if invalid and invalid(info):return "unparseable"
        positive=int(core[1:])%2==1
        wanted=("Rust / corrosion." if positive else "Normal / no defect.") if step=="q2" else (
            "A visible localized defect region." if positive else "No visible localized defect region.") if step=="presence" else (
            "Center region." if step=="position" else runner.old.NONE)
        return next(k for k,v in texts.items() if v==wanted)
    return infer


def run(args, calls, **kw):
    return runner.main(args, loader=lambda spec:{"kind":"internvl_chat","max_total_tiles":6}, infer=fake(calls,**kw))


def test_factorial_q1_fixed_actual_tiles_metrics_and_resume(tmp_path):
    rows,first,args,out,source=setup(tmp_path)
    before=source.read_bytes(); calls=[]
    assert run(args,calls)==0 and len(calls)==66
    answers=fixtures._csv(out/"answers.csv")
    assert len(answers)==6*3*5
    assert all(r["predicted_choice"]==first[r["id"]] for r in answers if r["question_type"]==runner.agent.Q1)
    assert not any("anomaly_detection" in r["question_type"] for r in runner.agent.read_jsonl(out/"predictions"/runner.MODEL/"steps.jsonl"))
    manifest=fixtures._csv(out/"image_manifest.csv")
    large=next(r for r in manifest if r["member"]=="sup/c1.png")
    assert (large["width"],large["standard_tiles"],large["zoom_tiles"])==("1080","1","5")
    assert all(r["standard_tiles"]==r["zoom_tiles"] for r in manifest if r["role"]=="reference")
    audit=runner.agent.read_jsonl(out/"prompts.jsonl")
    by={(r["question_id"],r["policy"],r["step"]):r for r in audit}
    for r in audit:
        if r["policy"] in ("control","prompt_only"):
            other="zoom_only" if r["policy"]=="control" else "zoom_prompt"
            assert r["prompt"]==by[r["question_id"],other,r["step"]]["prompt"]
    status=json.loads((out/"run_status.json").read_text())
    assert status["complete"] and all(r["q1_fixed"] for r in status["policies"])
    lb={r["policy"]:r for r in fixtures._csv(out/"leaderboard.csv")}
    assert len(lb)==5 and float(lb["control"]["q3_presence_balanced_accuracy"])==1
    assert all(r["q3_location_accuracy"]=="" and "q2_defect_balanced_accuracy" in r for r in lb.values())
    assert all("gold" not in r["scope"] for r in fixtures._csv(out/"significance.csv"))
    q3=fixtures._csv(out/"q3_by_defect.csv")
    assert next(r for r in q3 if r["policy"]=="cached_baseline" and r["defect"]=="rust")["missed_defects"]=="3"
    assert source.read_bytes()==before==(out/"cached_dev_baseline.jsonl").read_bytes()
    assert not (out/"frozen_config.json").exists() and not (out/"selection.csv").exists()
    calls.clear()
    assert runner.main(args,loader=lambda _:pytest.fail("resume loaded model"),infer=fake(calls))==0
    assert not calls and json.loads((out/"run_metadata.json").read_text())["reused_calls"]==66


@pytest.mark.parametrize("failure",["error","invalid"])
def test_failed_condition_excluded_and_only_failed_call_retried(tmp_path,failure):
    _,_,args,out,_=setup(tmp_path);calls=[]
    bad=lambda r:r[0]=="c0" and r[1]=="q2" and r[2]==[3,6] and "Normal metal need not" in r[3]
    assert run(args,calls,**{("fail" if failure=="error" else "invalid"):bad})==1
    lb=fixtures._csv(out/"leaderboard.csv")
    assert {r["policy"] for r in lb}==set(runner.POLICIES)-{"zoom_only"}
    assert not json.loads((out/"run_status.json").read_text())["complete"]
    assert all(r["condition"]!="zoom_only" and r["baseline"]!="zoom_only" for r in fixtures._csv(out/"significance.csv"))
    calls.clear()
    assert run(args,calls)==0 and len(calls)==1


def test_model_load_failure_is_attempted_once_and_never_eligible(tmp_path):
    _,_,args,out,_=setup(tmp_path);attempts=[]
    def loader(_):attempts.append(1);raise RuntimeError("cannot load fixture model")
    assert runner.main(args,loader=loader,infer=lambda *a:pytest.fail("infer after load failure"))==1
    assert len(attempts)==1
    status=json.loads((out/"run_status.json").read_text())
    assert not any(p["eligible_for_comparison"] for p in status["policies"])


def test_zoom_oom_blocks_only_zoom_view_and_retry_preserves_budget(tmp_path):
    _,_,args,out,_=setup(tmp_path);calls=[]
    regular=fake(calls)
    def infer(adapter,images,prompt):
        if adapter["image_tile_budgets"]==[3,6]:
            calls.append(("oom",))
            raise RuntimeError("CUDA out of memory")
        return regular(adapter,images,prompt)
    assert runner.main(args,loader=lambda _: {},infer=infer)==1
    assert calls.count(("oom",))==1
    assert {r["policy"] for r in fixtures._csv(out/"leaderboard.csv")}=={"cached_baseline","control","prompt_only"}
    records=runner.agent.read_jsonl(out/"predictions"/runner.MODEL/"steps.jsonl")
    assert all(r["image_tile_budgets"]==[3,6] for r in records if r["step"].startswith("zoom"))
    calls=[];assert run(args,calls)==0 and all(r[2]==[3,6] for r in calls)


def test_partial_extraction_can_resume_without_touching_source(tmp_path,monkeypatch):
    _,_,args,out,source=setup(tmp_path);before=source.read_bytes()
    real=runner.old.verified_images
    def interrupted(zip_path,members,root):
        if root.name=="normal":raise KeyboardInterrupt()
        return real(zip_path,members,root)
    monkeypatch.setattr(runner.old,"verified_images",interrupted)
    with pytest.raises(KeyboardInterrupt):run(args,[])
    assert json.loads((out/"run_metadata.json").read_text())["preflight_only"]
    monkeypatch.setattr(runner.old,"verified_images",real)
    assert run(args,[])==0 and source.read_bytes()==before


def test_interrupt_preserves_unvisited_cache_tail(tmp_path):
    _,_,args,out,_=setup(tmp_path);assert run(args,[])==0
    path=out/"predictions"/runner.MODEL/"steps.jsonl"
    records=runner.agent.read_jsonl(path)
    chosen=next(r for r in records if "zoom_only/q2" in r["id"] and r["question_id"].startswith("c0-"))
    chosen["error"]="retry me"
    runner.common.save_records(path,records)
    tail={r["id"]:r for r in records if r["question_id"].startswith("c5-")}
    with pytest.raises(KeyboardInterrupt):
        run(args,[],interrupt=lambda info:info[0]=="c0" and info[1]=="q2" and info[2]==[3,6])
    saved=runner.agent.read_jsonl(path)
    assert len(saved)==66 and {r["id"]:r for r in saved if r["id"] in tail}==tail
    assert not json.loads((out/"run_status.json").read_text())["complete"]
    calls=[];assert run(args,calls)==0 and len(calls)==1


def test_changed_inputs_and_invalid_cached_q1_do_not_overwrite_results(tmp_path):
    _,_,args,out,source=setup(tmp_path);assert run(args,[])==0
    before=(out/"answers.csv").read_bytes()
    records=runner.agent.read_jsonl(source);records[0]["output"]="INVALID"
    runner.common.save_records(source,records)
    with pytest.raises(ValueError,match="baseline failed/invalid"):run(args,[])
    assert (out/"answers.csv").read_bytes()==before
    records[0]["output"]=records[0]["predicted_choice"]
    records[0]["latency_s"]=99
    runner.common.save_records(source,records)
    with pytest.raises(ValueError,match="inputs/code/protocol changed"):run(args,[])
    assert (out/"answers.csv").read_bytes()==before


def test_prompts_never_use_gt_prior_answers_or_other_question_answers():
    rows=fixtures._rows();q2=next(r for r in rows if r["question_type"]==runner.agent.Q2)
    q3=runner.old.q3_rows([r for r in rows if r["question_type"]==runner.agent.Q1])[0]
    for short in (False,True):
        a=runner.prompts(q2,q3,short)
        b=runner.prompts({**q2,"answer":"D","anomaly_gt":"anomalous","defect_gt":"missing-cap"},
            {**q3,"answer":"C","anomaly_gt":"anomalous"},short)
        assert a[0]==b[0] and a[2:]==b[2:]
        assert "second look" not in a[0] and a[1]["options"]==runner.old.presence_task(q3)[0]["options"]


def test_no_gold_and_no_unrecognized_output_overwrite(tmp_path):
    rows,_,args,out,_=setup(tmp_path)
    qa=Path(args[args.index("--dev-qa")+1])
    runner.common.save_records(qa,[{**r,"protocol":"gold_v2_final"} for r in rows])
    with pytest.raises(ValueError,match="only gold_v2_dev"):run(args,[])
    runner.common.save_records(qa,rows)
    out.mkdir();sentinel=out/"existing.csv";sentinel.write_bytes(b"preserve")
    with pytest.raises(ValueError,match="unrecognized nonempty"):run(args,[])
    assert sentinel.read_bytes()==b"preserve"


def test_adapter_receives_exact_per_image_budgets_and_unchanged_reference(monkeypatch):
    hf=runner.hf; loads=[];chat=[]
    class Tensor:
        def __init__(self,n):self.n=n
        def size(self,i):return self.n
        def to(self,*a):return self
        def cuda(self):return self
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cat=lambda batches,dim:Tensor(sum(b.n for b in batches))))
    monkeypatch.setattr(hf,"_load_internvl_image",lambda path,max_num:(loads.append((str(path),max_num)) or Tensor(max_num)))
    def model_chat(*a,**kw):chat.append((a,kw));return "A"
    adapter={"max_total_tiles":6,"image_tile_budgets":[3,6],"dtype":"bf16","model":SimpleNamespace(chat=model_chat),"tokenizer":object()}
    assert hf._infer_internvl(adapter,[Path("ref.png"),Path("target.png")],"question")=="A"
    assert loads==[("ref.png",3),("target.png",6)] and chat[0][1]["num_patches_list"]==[3,6]
    assert chat[0][0][2].startswith("Image-1: <image>\nImage-2: <image>\n")
    assert chat[0][0][3]=={"max_new_tokens":80,"do_sample":False}
    assert hf._internvl_image_tile_budgets({"max_total_tiles":6},2)==[3,3]
    for bad in ([6],[3,0],[3,True],[3,1.5]):
        with pytest.raises(ValueError):hf._internvl_image_tile_budgets({**adapter,"image_tile_budgets":bad},2)


def notebook():
    return json.loads((runner.ROOT/"notebooks/lvlm_gold_v2_q23_zoom_prompt_dev_colab.ipynb").read_text("utf-8"))


def test_notebook_preflight_is_automatic_and_read_only(tmp_path):
    rows,_,args,out,baseline=setup(tmp_path)
    qa=Path(args[args.index("--dev-qa")+1]);refs=Path(args[args.index("--refs-csv")+1])
    dest=qa.parent/"refs/retrieved_refs.csv";dest.parent.mkdir();dest.write_bytes(refs.read_bytes())
    before={p:p.read_bytes() for p in (qa,baseline,dest)}
    context={"DRIVE_ROOT":tmp_path,"DEV_DIR":qa.parent,"STAGE1_DIR":baseline.parents[1],"RUN_ROOT":out,
        "json":json,"Path":Path,"N_IMAGES":6,"SEED":42,
        "SUP_ZIP_NAME":"sup.zip","UNSUP_ZIP_NAME":"unsup.zip",
        "SUP_ZIP_SIZE":(tmp_path/"sup.zip").stat().st_size,"UNSUP_ZIP_SIZE":(tmp_path/"unsup.zip").stat().st_size,
        "SUP_ZIP_PATH":"","UNSUP_ZIP_PATH":""}
    exec("".join(notebook()["cells"][3]["source"]),context)
    assert context["REFS_CSV"]==dest and len(context["rows"])==12
    assert all(p.read_bytes()==v for p,v in before.items()) and not out.exists()


def test_notebook_compile_display_and_zip(tmp_path,monkeypatch):
    _,_,args,out,_=setup(tmp_path);assert run(args,[])==0
    import pandas as pd
    nb=notebook()
    for i,c in enumerate(nb["cells"]):
        if c["cell_type"]=="code":compile("".join(c["source"]),f"cell{i}","exec")
    exec("".join(nb["cells"][5]["source"]),{"RUN_ROOT":out,"pd":pd,"json":json,"display":lambda x:None})
    downloads=[]
    monkeypatch.setitem(sys.modules,"google.colab",SimpleNamespace(files=SimpleNamespace(download=downloads.append)))
    archive=tmp_path/"result.zip"
    text="".join(nb["cells"][6]["source"]).replace('/content/dev_q23_zoom_prompt_results.zip',archive.as_posix())
    exec(text,{"RUN_ROOT":out,"Path":Path})
    with ZipFile(archive) as z:
        assert "run_metadata.json" in z.namelist() and "prompts.jsonl" in z.namelist()
        assert not any("_images" in n for n in z.namelist())
    assert downloads==[str(archive)]
