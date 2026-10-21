#!/usr/bin/env python3
"""Audit archived dev answers and create a compact, image-hash-bound input capsule."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import run_q2_conflict_evidence_dev as runner


def prepare(source, out):
    paths = [source/n for n in ('subset_qa_q12.jsonl','answers.csv','run_status.json','run_metadata.json',
        'image_manifest.csv','predictions/internvl3_5_8b/steps.jsonl')]
    source_sha = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    statuses = json.loads(paths[2].read_text('utf-8'))
    required = {r['policy']:r for r in statuses if r['model_key']==runner.MODEL}
    if any(not required.get(p,{}).get('eligible_for_comparison') or required[p].get('row_errors') or
           required[p].get('invalid_tool_answers') for p in ('baseline','previous_agent','q2_cues')):
        raise ValueError('historical source incomplete/ineligible')
    rows = runner.agent.read_jsonl(paths[0])
    answers = runner.read_csv(paths[1])
    by_id = {r['id']:r for r in rows}
    lookup = {}
    for saved in answers:
        if saved['policy'] not in ('baseline','previous_agent','q2_cues') or saved['id'] not in by_id:
            continue
        row = by_id[saved['id']]
        key = saved['policy'],saved['id']
        if key in lookup or any(saved.get(k)!=row.get(k) for k in ('core_id','question_type','asset_category','answer')) or saved['model_key']!=runner.MODEL:
            raise ValueError('duplicate/mismatched saved answer')
        lookup[key] = saved['predicted_choice']
    if len(lookup)!=3*len(rows):
        raise ValueError('missing historical answers')
    first = {rid:lookup['baseline',rid] for rid in by_id}
    steps = runner.agent.read_jsonl(paths[5])
    reviewed, cues = {}, {}
    for step in steps:
        if step['step']=='q2_define':
            rid = step['question_id']
            if rid in reviewed or rid not in by_id or lookup['previous_agent',rid]!=step['predicted_choice']:
                raise ValueError('duplicate/mismatched raw review')
            reviewed[rid] = step
        if step['step']=='q2_cues':
            rid=step['question_id']
            if rid in cues or rid not in by_id or lookup['q2_cues',rid]!=step['predicted_choice']:
                raise ValueError('duplicate/mismatched cue answer')
            cues[rid]=step
    for row in rows:
        if row['question_type']==runner.agent.Q1 and lookup['previous_agent',row['id']]!=first[row['id']]:
            raise ValueError('historical Q1 changed')
    references = {}
    for rid,step in reviewed.items():
        marker = '/_images/normal/'
        if marker not in step['images'][0]:
            raise ValueError('unrecognized normal reference path')
        references[by_id[rid]['image']] = step['images'][0].split(marker,1)[1]
    manifest = [{k:r[k] for k in ('role','member','sha256')} for r in runner.read_csv(paths[4])]
    data = {'protocol':runner.PROTOCOL,'rows':rows,'initial':first,'reviewed':reviewed,'cue_records':cues,
        'references':references,'image_manifest':manifest,'source_sha256':source_sha,
        'source_model_revision':'historical revision not recorded',
        'provenance':'archived refined dev baseline and exact q2_define raw outputs; no new inference'}
    runner.validate_dataset(data)
    _,cohorts = runner.refined.select_rows(rows,len(rows)//2,42)
    table = []
    for name,cores in cohorts.items():
        table.extend({'cohort':name,**r} for r in runner.transition_report([r for r in rows if r['core_id'] in cores],first,reviewed))
    details = []
    for r in rows:
        if r['question_type']==runner.agent.Q2 and first[r['id']]!=reviewed[r['id']]['predicted_choice']:
            details.append({'id':r['id'],'core_id':r['core_id'],'asset_category':r['asset_category'],
                'defect_gt':r['defect_gt'],'answer':r['answer'],'initial':first[r['id']],
                'review':reviewed[r['id']]['predicted_choice'],
                'initial_correct':first[r['id']]==r['answer'],
                'review_correct':reviewed[r['id']]['predicted_choice']==r['answer']})
    out.mkdir(parents=True,exist_ok=True)
    dataset = out/'dataset.json'
    payload = (json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode()
    if dataset.exists() and dataset.read_bytes()!=payload:
        raise ValueError('prepared dataset changed; use new output directory')
    dataset.write_bytes(payload)
    runner.agent.cons.write_csv(out/'historical_transitions.csv',table)
    runner.agent.cons.write_csv(out/'historical_conflicts.csv',details)
    summary = {'dataset_sha256':hashlib.sha256(payload).hexdigest(),'source_sha256':source_sha,
        'images':len(rows)//2,'transitions':table,'q1_unchanged':True,'new_model_calls':0}
    runner.old.atomic_json(out/'audit_summary.json',summary)
    lines = ['# Q2 첫 판단 → 정의 재검토: CPU 분석','',
        '기존 dev240 응답을 검산했다. 새 모델 추론0회. gold/Q1 변경0. 정답은 사후 분석에만 사용한다.',
        '초기 판단과 정의 재검토의 차이는 프롬프트를 포함한 전체 조건 차이이며 내부 사고 원인을 설명하지 않는다.','',
        '| 범위 | 유형 | 장수 | 충돌 | 정답→오답 | 오답→정답 | 다른 오답 | 초기 정답 | 재검토 정답 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in table:
        lines.append('| '+' | '.join(str(t[k]) for k in ('cohort','defect','images','disagreements',
            'correct_to_wrong','wrong_to_correct','wrong_to_other_wrong','initial_correct','review_correct'))+' |')
    design = next(t for t in table if t['cohort']=='design_dev40' and t['defect']=='ALL')
    lines += ['','## 다음 실행',
        f"고정 design40의 충돌{design['disagreements']}건 × (후보별 확인2 + 대조 재질문2) = 신규{4*design['disagreements']}호출.",
        '후보별 이진 지지 확인은 새 탐색 가설이다. 둘 다 지지/둘 다 미지지는 초기 답 유지. 무효 출력은 유지하고 해당 정책을 비교에서 제외한다.',
        '초기 답·정의 재검토·기존 cue 답을 자동 재실행하지 않는다. 이미 더 강했던 cue 조건도 비교하며 예전 초기 답보다 개선됐다는 이유만으로 채택하지 않는다. 기존 dev는 이미 분석했으므로 독립 최종검증으로 부르지 않는다.',
        '40장 결과와 정답 퇴행·결함별 미탐·무효·호출시간을 확인한 다음 별도 추가200 실행 여부를 정한다.',
        '실제 새 구조의 효과는 아직 미검증. 새 GPU 실행 및 GitHub 게시 없음.']
    (out/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'images':summary['images'],'dataset_sha256':summary['dataset_sha256'],
        'transitions':[t for t in table if t['defect']=='ALL'],'pilot_new_calls':4*design['disagreements']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-results',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args()
    prepare(args.source_results,args.out_dir)
