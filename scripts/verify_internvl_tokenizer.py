"""Verify native pinned Qwen tokenizer without applying a Mistral regex patch."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import logging

REVISION='9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79'
FILES={
    'added_tokens.json':'e5c2531bd62d25dd2a7401203a20f7ffb0976a026299cb4d6ff2d559d7ac3e28',
    'merges.txt':'8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
    'special_tokens_map.json':'13a6dea522a937ef05172912c66214162c75165e723a3d6cf4ae2d09752013bf',
    'vocab.json':'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
    'config.json':'e849152d81a9eb158475d493d6c5ba1fc935d35ed01429bbd0182582416bb963',
    'tokenizer.json':'6581c44164d273d4222df982905a7e0450dcf3a4a7ebe98f9ec53e4de05beffe',
    'tokenizer_config.json':'01fa09d7bd718d09a7d795aec4fb310f2be928b5273fd16faf64037f0e3993d5',
}
sha=lambda b:hashlib.sha256(b).hexdigest()
digest=lambda x:sha(json.dumps(x,sort_keys=True,ensure_ascii=False).encode())


def probes(prompt):
    fixed=[prompt,'{"present":true}','{"present":false}','{"regions":[]}',
           '<image>','<IMG_CONTEXT>','<img>','</img>',
           '<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n',
           '1234567890 2026-10-21 3.14159','HTTPServer XMLParser McDonald USA',
           "I'm don't we're they've she'll",'누락 유리 절연체 缺失 insulator component',
           'A/B\n\t whitespace   and trailing ',
           'Image-1: <image>\nImage-2: <image>\nImage-3: <image>\n'+prompt]
    return fixed+[f'glassDisc{i} MissingCAP_{i}/hardware {i:06d}\n' for i in range(64)]


def verify(snapshot,prompt,*,tokenizer=None):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    snapshot=Path(snapshot)
    if importlib.metadata.version('transformers')!='5.14.0':raise ValueError('pinned transformers5.14.0 required')
    for n,h in FILES.items():
        if sha((snapshot/n).read_bytes())!=h:raise ValueError('pinned tokenizer source changed: '+n)
    cfg=json.loads((snapshot/'config.json').read_text());tcfg=json.loads((snapshot/'tokenizer_config.json').read_text())
    if cfg['model_type']!='internvl_chat' or cfg['llm_config']['model_type']!='qwen3' or tcfg['tokenizer_class']!='Qwen2Tokenizer':
        raise ValueError('not the audited Qwen-family model')
    warnings=[]
    class Capture(logging.Handler):
        def emit(self,record):warnings.append(record.getMessage())
    logger=logging.getLogger('transformers');handler=Capture();logger.addHandler(handler)
    try:
        # Same native call as previous adapter. No fix flag, warning filter or model loaded.
        if tokenizer is None:tokenizer=AutoTokenizer.from_pretrained(str(snapshot),trust_remote_code=True,use_fast=False,local_files_only=True)
    finally:logger.removeHandler(handler)
    backend=getattr(tokenizer,'backend_tokenizer',getattr(tokenizer,'_tokenizer',None))
    if backend is None:raise ValueError('native backend missing')
    raw=Tokenizer.from_file(str(snapshot/'tokenizer.json'))
    canonical=json.loads(raw.to_str());actual=json.loads(backend.to_str())
    for k in sorted(set(canonical)|set(actual)):
        if canonical.get(k)!=actual.get(k):raise ValueError('checkpoint tokenizer mismatch: '+k)
    cases=probes(prompt);tokens=[]
    for text in cases:
        for add in [False,True]:
            ids=tokenizer.encode(text,add_special_tokens=add)
            if ids!=raw.encode(text,add_special_tokens=add).ids:raise ValueError('native encoding differs from checkpoint')
            tokens.append(ids)
    for text in ['<IMG_CONTEXT>','<img>','</img>','<|im_start|>','<|im_end|>']:
        if tokenizer.convert_tokens_to_ids(text)!=raw.token_to_id(text):raise ValueError('special token changed: '+text)
    receipt=dict(protocol='internvl_native_tokenizer_equivalence_v1',complete=True,model_revision=REVISION,
        source_files=FILES,model_family='qwen3',tokenizer_class=type(tokenizer).__name__,
        transformers_version=importlib.metadata.version('transformers'),tokenizers_version=importlib.metadata.version('tokenizers'),
        pre_tokenizer_sha256=digest(actual['pre_tokenizer']),backend_sha256=digest(actual),
        probe_count=len(cases),encoding_checks=len(tokens),probe_ids_sha256=digest(tokens),
        source_backend_equal=True,encoding_equal=True,special_tokens_equal=True,
        regex_patch_applied=bool(getattr(backend,'fix_mistral_regex',False)),original_warnings=warnings,
        interpretation='native encoding matches publisher checkpoint; local Mistral warning is not evidence of wrong Qwen encoding',
        neural_calls=0)
    if receipt['regex_patch_applied']:raise ValueError('unexpected Mistral regex mutation')
    return tokenizer,receipt
