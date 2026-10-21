#!/usr/bin/env python3
"""Dev-only binary candidate support versus two matched sequential reasks."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import run_gold_v2_q23_zoom_prompt_dev as previous

agent, old, refined, common, hf = previous.agent, previous.old, previous.refined, previous.common, previous.hf
MODEL = previous.MODEL
MODEL_ID = 'OpenGVLab/InternVL3_5-8B'
MODEL_REVISION = '9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79'
PROTOCOL = 'q2_conflict_binary_evidence_dev_v1'
PREFIX = 'q2_conflict_evidence_dev'
POLICIES = ('initial', 'definition_review', 'cached_cues', 'retain_initial', 'matched_reask', 'candidate_support')
COMPARISONS = [('initial', 'candidate_support'), ('definition_review', 'candidate_support'),
               ('cached_cues', 'candidate_support'), ('matched_reask', 'candidate_support')]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def strict_choice(text, options):
    # One letter, optionally its exact option text; reject prose/multiple answers.
    text = text.strip()
    if text in options:
        return text
    for letter, description in options.items():
        if text == f'{letter}. {description}':
            return letter
    return None


def validate_dataset(data):
    rows = data['rows']
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)) or not rows:
        raise ValueError('duplicate/empty QA')
    if any(r['question_type'] not in (agent.Q1,agent.Q2) for r in rows):
        raise ValueError('only paired Q1/Q2 dev rows allowed')
    # Reuse existing paired-dev validation and fixed design40 selector.
    refined.select_rows(rows, len(rows)//2, 42)
    first, reviewed = data['initial'], data['reviewed']
    if set(first) != set(ids):
        raise ValueError('missing/extra initial Q1/Q2 answers')
    q2 = [r for r in rows if r['question_type'] == agent.Q2]
    if set(reviewed) != {r['id'] for r in q2}:
        raise ValueError('missing/extra reviewed Q2 answers')
    if set(data['cue_records']) != set(reviewed):
        raise ValueError('missing current cue comparator')
    for row in rows:
        if first[row['id']] not in row['options']:
            raise ValueError('invalid initial choice')
        if row['question_type'] == agent.Q2:
            rec = reviewed[row['id']]
            if (rec.get('question_id'), rec.get('question_type'), rec.get('model_key'), rec.get('step')) != (
                    row['id'], agent.Q2, MODEL, 'q2_define'):
                raise ValueError('review identity mismatch')
            if (rec.get('error') or rec.get('predicted_choice') not in row['options'] or
                    agent.base.parse_choice(rec.get('output', ''), set(row['options'])) != rec['predicted_choice'] or
                    rec.get('prompt') != agent.build_prompt(row, first[row['id']], 'define')):
                raise ValueError('review raw/prompt/choice mismatch')
            ref = data['references'].get(row['image'])
            if not ref or len(rec.get('images', [])) != 2 or not rec['images'][0].endswith('/'+ref) or not rec['images'][1].endswith('/'+row['image']):
                raise ValueError('historical image/reference order mismatch')
            cue = data['cue_records'][row['id']]
            if (cue.get('question_id'),cue.get('question_type'),cue.get('model_key'),cue.get('step')) != (
                    row['id'],agent.Q2,MODEL,'q2_cues') or cue.get('error') or cue.get('images')!=rec['images'] or (
                    cue.get('prompt')!=refined.definition_prompt(row,True) or
                    cue.get('predicted_choice') not in row['options'] or
                    agent.base.parse_choice(cue.get('output',''),set(row['options']))!=cue['predicted_choice']):
                raise ValueError('cue comparator identity/raw/prompt mismatch')
    expected = {(r['role'], r['member']): r['sha256'] for r in data['image_manifest']}
    if len(expected) != len(data['image_manifest']):
        raise ValueError('duplicate image manifest')
    for row in q2:
        for key in [('target', row['image']), ('reference', data['references'][row['image']])]:
            if not re.fullmatch('[0-9a-f]{64}', expected.get(key, '')):
                raise ValueError('missing image SHA256')
    return data


def support_task(row, candidate):
    if row['question_type'] != agent.Q2 or candidate not in row['options']:
        raise ValueError('valid same-Q2 candidate required')
    text = row['options'][candidate]
    options = {'A': 'Supported by visible evidence.', 'B': 'Not supported by visible evidence.'}
    if int(hashlib.sha256((row['id']+':'+text).encode()).hexdigest(), 16) % 2:
        options = {'A': options['B'], 'B': options['A']}
    prompt = ('You are an industrial visual inspector. Image 1 is a verified NORMAL reference of the same asset '
              'category. Image 2 is the TARGET. Judge only the TARGET; the reference can differ in pose and lighting.\n\n'
              f'Candidate condition: {text}\nVisual criterion: {refined.DEFINITIONS[text]}\n\n'
              'Check this condition independently. The candidate is a hypothesis, not a previous observation. '
              'Select supported only when the TARGET visibly meets its criterion. A lack of clear evidence '
              'does not establish the competing condition. Do not infer damage from viewpoint or background alone.\n'
              + '\n'.join(f'{k}. {v}' for k,v in options.items()) +
              '\n\nReturn exactly one option letter and nothing else.')
    return {**row, 'options': options}, prompt


def resolve_support(initial, review, initial_support, review_support):
    if initial_support is None or review_support is None:
        return None
    if initial_support != review_support:
        return initial if initial_support else review
    return initial


def transition_report(rows, initial, reviewed):
    output = []
    for defect in ['ALL', *sorted({r['defect_gt'] for r in rows if r['question_type']==agent.Q2})]:
        subset = [r for r in rows if r['question_type']==agent.Q2 and (defect=='ALL' or r['defect_gt']==defect)]
        counts = Counter()
        for r in subset:
            a,b = initial[r['id']], reviewed[r['id']]['predicted_choice']
            counts['disagreements'] += a != b
            counts['correct_to_wrong'] += a==r['answer'] and b!=r['answer']
            counts['wrong_to_correct'] += a!=r['answer'] and b==r['answer']
            counts['wrong_to_other_wrong'] += a!=r['answer'] and b!=r['answer'] and a!=b
            counts['initial_correct'] += a==r['answer']
            counts['review_correct'] += b==r['answer']
        output.append({'defect':defect, 'images':len(subset), **{k:counts[k] for k in (
            'disagreements','correct_to_wrong','wrong_to_correct','wrong_to_other_wrong','initial_correct','review_correct')}})
    return output


def code_hashes():
    paths = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.suffix=='.py' and path.relative_to(ROOT).parts[0] in ('scripts','src'):
                paths.add(path)
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main(argv=None, *, loader=None, infer=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','sup-zip','unsup-zip','out-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--n-images', type=int, default=40)
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--model-path', type=Path)
    args = parser.parse_args(argv)
    out = args.out_dir
    if not out.name.startswith(PREFIX) or args.n_boot < 1:
        raise ValueError('use separate q2_conflict_evidence_dev* output and positive n-boot')
    if (out.resolve().is_relative_to(args.dataset.parent.resolve()) or
        any(out.resolve()==p.resolve() or p.resolve().is_relative_to(out.resolve())
            for p in (args.dataset,args.sup_zip,args.unsup_zip))):
        raise ValueError('output overlaps source directory')
    data = validate_dataset(json.loads(args.dataset.read_text(encoding='utf-8')))
    rows, cohorts = refined.select_rows(data['rows'], args.n_images, 42)
    first = {r['id']:data['initial'][r['id']] for r in rows}
    q2 = [r for r in rows if r['question_type']==agent.Q2]
    reviewed = {r['id']:data['reviewed'][r['id']]['predicted_choice'] for r in q2}
    routed = [r for r in q2 if first[r['id']] != reviewed[r['id']]]
    spec = {**agent.base.MODEL_REGISTRY[MODEL]}
    if loader is None:
        if args.model_path is None or args.model_path.name != MODEL_REVISION or not args.model_path.is_dir():
            raise ValueError('native run needs the pinned Hugging Face snapshot directory')
        spec['model_id'] = str(args.model_path)
    identity_base = {'protocol':PROTOCOL,'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        'rows':rows,'code_sha256':code_hashes(),'model_id':MODEL_ID,'model_revision':MODEL_REVISION,
        'backend':'hf' if loader is None else 'fixture','n_boot':args.n_boot,
        'generation':{'max_new_tokens':80,'do_sample':False,'image_tile_budgets':[3,3]},
        'model_path':str(args.model_path) if args.model_path else None}
    signature = digest(identity_base)
    mp = out/'run_metadata.json'
    prev = json.loads(mp.read_text('utf-8')) if mp.exists() else {}
    if prev and prev.get('input_signature') != signature:
        raise ValueError('inputs/code/model changed; use a new output directory')
    if not prev and out.exists() and any(out.iterdir()):
        raise ValueError('unrecognized nonempty output directory')
    targets = old.verified_images(args.sup_zip,[r['image'] for r in q2],out/'_images/target')
    refs = {r['image']:data['references'][r['image']] for r in q2}
    normals = old.verified_images(args.unsup_zip,list(refs.values()),out/'_images/normal')
    manifest = previous.tile_manifest(targets,normals)
    expected = {(r['role'],r['member']):r['sha256'] for r in data['image_manifest']}
    if any(expected.get((r['role'],r['member'])) != r['sha256'] for r in manifest):
        raise ValueError('source image bytes differ from historical SHA256')
    identity = digest([signature,manifest])
    if prev.get('prediction_identity') and prev['prediction_identity']!=identity:
        raise ValueError('image identity changed')
    meta = {**identity_base,'input_signature':signature,'prediction_identity':identity,'complete':False,
        'q1_inference_calls':0,'q3_evaluated':False,'gold_used':False,
        'historical_revision':'not recorded; cached first/review provenance retained; no fresh end-to-end claim',
        'routed_questions':len(routed),'planned_new_calls':4*len(routed),
        'deployment_cost':'two cached Q2 readings/image + two calls/conflict per policy; cache reuse is not free deployment',
        'claim_limit':'already examined development data; binary self-support is not independent truth or generalization'}
    old.atomic_json(mp,meta)
    old.atomic_json(out/'run_status.json',{'status':'running','complete':False})
    (out/'dataset_snapshot.json').write_bytes(args.dataset.read_bytes())
    common.save_records(out/'subset_qa_q12.jsonl',rows)
    agent.cons.write_csv(out/'image_manifest.csv',manifest)
    agent.cons.write_csv(out/'cohort_manifest.csv',[{'core_id':r['core_id'],'cohort':'design_dev40' if r['core_id'] in cohorts['design_dev40'] else 'additional_dev'} for r in q2])
    path = out/'predictions'/MODEL/'steps.jsonl'
    records = agent.read_jsonl(path) if path.exists() else []
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate cached step IDs')
    existing, used = {r['id']:r for r in records}, {}
    counts = Counter(called_this_run=0,reused_calls=0,blocked_calls=0)
    if loader is None:
        def native_loader(model_spec):
            import torch
            if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 20*1024**3:
                raise RuntimeError('native HF requires L4/A100 VRAM >=20GB')
            return hf.load_hf_adapter(model_spec)
        loader = native_loader
    holder = agent.base.LazyHFAdapter(spec,loader=loader)
    infer = hf.infer_hf if infer is None else infer
    load_failure = ''

    def save():
        common.save_records(path,[*used.values(),*(r for k,r in existing.items() if k not in used)])

    def call(row, step, prompt, dependency_error=''):
        nonlocal load_failure
        key = row['id']+'/'+step
        images = [normals[refs[row['image']]],targets[row['image']]]
        request = {'identity':identity,'prompt':prompt,'options':row['options'],'images':list(map(str,images)),
                   'image_tile_budgets':[3,3]}
        request_hash = digest(request)
        rec = existing.get(key)
        if rec and rec.get('error'):
            old.atomic_json(out/'attempt_history'/(digest(rec)+'.json'),rec)
        if rec and (rec.get('request_sha256')!=request_hash or any(rec.get(k)!=v for k,v in request.items())):
            # A failed first reask can recover to a different valid answer on resume.
            # Its blocked dependent call has never run; preserve that record and make the new request.
            if step=='matched_reask_2' and rec.get('error','').startswith('BlockedDependency:') and rec.get('identity')==identity:
                old.atomic_json(out/'attempt_history'/(digest(rec)+'.json'),rec)
                rec=None
            else:
                raise ValueError('cached request mismatch')
        if rec and not rec.get('error'):
            if strict_choice(rec['output'],row['options']) != rec.get('predicted_choice'):
                raise ValueError('cached parse mismatch')
            counts['reused_calls'] += 1  # Invalid outputs are retained, never correctness-retried.
        else:
            output,error,start = '',dependency_error or load_failure,time.time()
            if not error:
                try:
                    adapter = holder.get()
                except Exception as exc:
                    load_failure = error = f'{type(exc).__name__}: {exc}'
                if not error:
                    try:
                        output = str(infer({**adapter,'image_tile_budgets':[3,3]},images,prompt))
                    except Exception as exc:
                        error = f'{type(exc).__name__}: {exc}'
                        if 'out of memory' in error.lower():
                            load_failure = error
                counts['called_this_run'] += 1
            else:
                counts['blocked_calls'] += 1
            rec = {'id':key,'question_id':row['id'],'question_type':agent.Q2,'step':step,'model_key':MODEL,
                **request,'request_sha256':request_hash,'output':output,'error':error,
                'predicted_choice':strict_choice(output,row['options']),'latency_s':round(time.time()-start,3)}
        used[key] = rec
        save()  # Every call persisted, including invalid/error.
        if len(used)%20==0:
            old.atomic_json(out/'run_status.json',{'status':'running','complete':False,'finished_calls':len(used),'planned_calls':4*len(routed)})
            print(f'{len(used)}/{4*len(routed)} calls saved',flush=True)
        return rec

    finals = {p:dict(first) for p in POLICIES}
    finals['definition_review'].update(reviewed)
    finals['cached_cues'].update({r['id']:data['cue_records'][r['id']]['predicted_choice'] for r in q2})
    deps = {p:[] for p in POLICIES}
    decisions = []
    try:
        for row in routed:
            rid = row['id']
            a,b = first[rid],reviewed[rid]
            # Same two calls per conflict. Alternate policy/candidate order by ID, without labels.
            outputs = {}
            order = ['support','reask']
            if int(hashlib.sha256(rid.encode()).hexdigest(),16)%2:
                order.reverse()
            for kind in order:
                if kind=='support':
                    candidates = [('initial',a),('review',b)]
                    if int(hashlib.sha256((rid+':order').encode()).hexdigest(),16)%2:
                        candidates.reverse()
                    for label,candidate in candidates:
                        task,prompt = support_task(row,candidate)
                        rec = call(task,'support_'+label,prompt)
                        deps['candidate_support'].append(rec)
                        outputs[label] = None if rec['predicted_choice'] is None else task['options'][rec['predicted_choice']]=='Supported by visible evidence.'
                else:
                    rec1 = call(row,'matched_reask_1',agent.build_prompt(row,b,'reask'))
                    # Invalid first reask is not silently replaced by an old answer.
                    rec2 = call(row,'matched_reask_2',agent.build_prompt(row,rec1['predicted_choice'],'reask'),
                        'BlockedDependency: matched_reask_1 failed' if rec1['error'] else '')
                    deps['matched_reask'].extend([rec1,rec2])
                    finals['matched_reask'][rid] = rec2['predicted_choice'] if rec1['predicted_choice'] else None
            finals['candidate_support'][rid] = resolve_support(a,b,outputs['initial'],outputs['review'])
            decisions.append({'id':rid,'initial':a,'review':b,'initial_supported':outputs['initial'],
                'review_supported':outputs['review'],'final':finals['candidate_support'][rid]})
    except BaseException:
        old.atomic_json(out/'run_status.json',{'status':'interrupted','complete':False,'finished_calls':len(used)})
        raise
    finally:
        save()
        holder.release()
    if len(used)!=4*len(routed) or sum(counts.values())!=4*len(routed):
        raise RuntimeError('planned/called/reused/blocked call coverage mismatch')
    answers,leaderboard,by_defect,by_asset,statuses,found,cohort_metrics,changes = [],[],[],[],[],{},[],[]
    for policy in POLICIES:
        errors = sum(bool(r['error']) for r in deps[policy])
        invalids = sum(r['predicted_choice'] is None for r in deps[policy])
        q1_fixed = all(finals[policy][r['id']]==first[r['id']] for r in rows if r['question_type']==agent.Q1)
        eligible = not errors and not invalids and q1_fixed
        statuses.append({'policy':policy,'eligible_for_comparison':eligible,'q1_fixed':q1_fixed,'row_errors':errors,
            'invalid_tool_answers':invalids,'additional_calls':len(deps[policy])})
        for row in rows:
            answers.append({'policy':policy,'id':row['id'],'core_id':row['core_id'],'question_type':row['question_type'],
                'asset_category':row['asset_category'],'answer':row['answer'],'initial_choice':first[row['id']],
                'review_choice':reviewed.get(row['id']),'predicted_choice':finals[policy][row['id']],
                'eligible_for_comparison':eligible})
        # Diagnostic counts retained even for excluded policies, with eligibility flag.
        for label,cores in cohorts.items():
            subset = [r for r in rows if r['core_id'] in cores]
            if subset:
                cohort_metrics.append({'policy':policy,'cohort':label,'eligible_for_comparison':eligible,
                    **common.q12_metrics(subset,finals[policy])})
        q2choices = {rid:{'predicted_choice':choice} for rid,choice in finals[policy].items()}
        changes.extend({'policy':policy,'eligible_for_comparison':eligible,**r} for r in transition_report(rows,first,q2choices))
        if not eligible:
            continue
        found[MODEL,policy] = finals[policy]
        leaderboard.append({'policy':policy,**common.q12_metrics(rows,finals[policy]),'additional_calls':len(deps[policy])})
        for asset in sorted({r['asset_category'] for r in q2}):
            by_asset.append({'policy':policy,'asset_category':asset,**common.q12_metrics([r for r in rows if r['asset_category']==asset],finals[policy])})
        for defect in sorted({r['defect_gt'] for r in q2}):
            subset = [r for r in q2 if r['defect_gt']==defect]
            by_defect.append({'policy':policy,'defect':defect,'images':len(subset),
                'correct':sum(finals[policy][r['id']]==r['answer'] for r in subset),
                'false_normal':sum(row['anomaly_gt']=='anomalous' and agent.base.predicted_semantic(row,finals[policy][row['id']])=='normal' for row in subset)})
    significance = [dict(r,scope='exploratory previously examined dev; Q2 only')
        for r in agent.cons.significance_rows(agent.cons.image_pairs(rows),[],found,[MODEL],COMPARISONS,args.n_boot)
        if r['metric'] not in ('q3_accuracy','q1q3_contradiction')]
    for name,table in [('answers.csv',answers),('leaderboard.csv',leaderboard),('q2_by_defect.csv',by_defect),
            ('metrics_by_asset.csv',by_asset),('cohort_metrics.csv',cohort_metrics),('transitions.csv',changes),('significance.csv',significance)]:
        agent.cons.write_csv(out/name,table)
    old.atomic_json(out/'decisions.json',decisions)
    complete = all(s['eligible_for_comparison'] for s in statuses)
    meta.update({'complete':complete,**counts,'raw_calls':len(used),'gpu':agent.gpu_info(),
        'row_errors':sum(bool(r['error']) for r in used.values()),
        'invalid_tool_answers':sum(r['predicted_choice'] is None for r in used.values()),
        'latency_s_by_policy':{p:sum(r['latency_s'] for r in deps[p]) for p in POLICIES}})
    meta['software']={}
    for package in ('torch','transformers','accelerate','pillow','numpy','timm'):
        try:
            meta['software'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            meta['software'][package]=None
    old.atomic_json(mp,meta)
    old.atomic_json(out/'run_status.json',{'status':'complete' if complete else 'ineligible','complete':complete,
        'policies':statuses,'selection_performed':False,'result_published':False})
    print(json.dumps({'complete':complete,'routed_questions':len(routed),'planned_calls':4*len(routed),**counts}),flush=True)
    return 0 if complete else 1


if __name__=='__main__':
    raise SystemExit(main())
