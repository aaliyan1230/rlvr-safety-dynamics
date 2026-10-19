#!/usr/bin/env python3
"""Pin public metadata and validate every legacy prompt without paid inference."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rlvr_safety.checkpoint_protocol import format_controls  # noqa: E402
from rlvr_safety.checkpoint_study import (  # noqa: E402
    THINK_STEPS,
    atomic_json,
    build_matrix,
    coverage_ids,
    digest,
)
from rlvr_safety.io import sha256_file, write_jsonl  # noqa: E402

USER_AGENT = 'OpenAI File Downloader, XaiImageApiFetch/1.0'
FILES = ('config.json', 'generation_config.json', 'tokenizer.json',
         'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja')


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def metadata(step):
    repo = 'allenai/Olmo-3-7B-Think' + ('-DPO' if step == 0 else '')
    branch = 'main' if step == 0 else f'step_{step:04d}'
    info = json.loads(fetch(f'https://huggingface.co/api/models/{repo}/revision/{branch}?blobs=true'))
    weights = {s['rfilename']: s['lfs']['sha256'] for s in info['siblings']
               if s['rfilename'].endswith('.safetensors')}
    names = {s['rfilename'] for s in info['siblings']}
    if not weights or not all(len(h) == 64 for h in weights.values()):
        raise ValueError('missing weight hashes')
    return {'step': step, 'repo': repo, 'branch': branch, 'revision': info['sha'],
            'weights': weights, 'siblings': info['siblings'],
            'tokenizer_blobs': {s['rfilename']: s['blobId'] for s in info['siblings']
                                if s['rfilename'] in FILES},
            'available_files': sorted(names)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'FAIL', 'paid_inference': False}
    try:
        matrix = build_matrix(ROOT)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            models = list(pool.map(metadata, THINK_STEPS))
        if len({digest(m['weights']) for m in models}) != 12:
            raise ValueError('checkpoint weights are not distinct')
        for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
            if len({m['tokenizer_blobs'].get(name) for m in models}) != 1:
                raise ValueError(f'{name} differs across checkpoints')
            if name not in models[0]['tokenizer_blobs']:
                raise ValueError(f'missing native {name}')
        by_step = {m['step']: m for m in models}
        for row in matrix:
            row['revision'] = by_step[row['step']]['revision']
            row['repo'] = by_step[row['step']]['repo']
            row['input_hash'] = digest({k: v for k, v in row.items() if k != 'input_hash'})
        template_dir = args.out / 'tokenizer'
        template_dir.mkdir(exist_ok=True)
        model = models[0]
        for name in FILES:
            if name in model['available_files']:
                (template_dir / name).write_bytes(fetch(
                    f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{name}"))
        from transformers import AutoTokenizer, GenerationConfig

        from rlvr_safety.permission.generation import build_effective_config
        tokenizer = AutoTokenizer.from_pretrained(template_dir, local_files_only=True,
                                                  trust_remote_code=False)
        if not tokenizer.chat_template:
            raise ValueError('no native chat template; generic fallback forbidden')
        config = json.loads((template_dir / 'config.json').read_text())
        context = config['max_position_embeddings']
        cap = 32768  # Validate the permitted repair allowance too.
        rendered_hashes, input_lengths = {}, []
        rendered_inputs = [*matrix, *format_controls(matrix[0]), *format_controls(matrix[-1])]
        for row in rendered_inputs:
            messages = [{'role': 'system', 'content': row['system_prompt']},
                        {'role': 'user', 'content': row['prompt']}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
            ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                                add_generation_prompt=True)
            if tokenizer.encode(rendered, add_special_tokens=False) != ids:
                raise ValueError('rendering/tokenization disagreement')
            if len(ids) + cap > context:
                raise ValueError('prompt plus allowed output exceeds native context')
            rendered_hashes[row['id']] = digest(rendered)
            input_lengths.append(len(ids))
        inherited = GenerationConfig.from_pretrained(template_dir, local_files_only=True)
        eos = inherited.eos_token_id or tokenizer.eos_token_id
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
            eos[0] if isinstance(eos, list) else eos)
        _, effective, unused = build_effective_config(inherited, 8192, pad)
        if effective.get('forced_eos_token_id') is not None or effective['do_sample']:
            raise ValueError('generation overrides failed')
        generation_reports = []
        for model in models:
            url = (f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/"
                   'generation_config.json')
            inherited_model = GenerationConfig.from_dict(json.loads(fetch(url)))
            _, resolved, _ = build_effective_config(inherited_model, 8192, pad)
            additional_expected = {'sequence_bias': None, 'forced_decoder_ids': None,
                'force_words_ids': None, 'constraints': None, 'stop_strings': None,
                'max_time': None, 'encoder_repetition_penalty': 1.0, 'num_return_sequences': 1}
            if any(resolved.get(k, default) != default
                   for k, default in additional_expected.items()):
                raise ValueError('inherited transforms/stops differ from the intended generation')
            native_eos = resolved.get('eos_token_id') or tokenizer.eos_token_id
            if native_eos != eos:
                raise ValueError('native EOS changes across checkpoints; investigate before GPU')
            generation_reports.append({'step': model['step'], 'revision': model['revision'],
                                       'effective_config_hash': digest(resolved),
                                       'eos': native_eos})
        release = json.loads(fetch('https://huggingface.co/api/models/allenai/Olmo-3-7B-Think?blobs=true'))
        release_weights = {s['rfilename']: s['lfs']['sha256'] for s in release['siblings']
                           if s['rfilename'].endswith('.safetensors')}
        if release_weights == models[-1]['weights']:
            raise ValueError('release/last-numbered distinction changed; inspect protocol')
        recipe_url = ('https://raw.githubusercontent.com/allenai/open-instruct/main/'
                      'scripts/train/olmo3/7b_think_rl.sh')
        (args.out / '7b_think_rl.sh').write_bytes(fetch(recipe_url))
        write_jsonl(args.out / 'matrix.jsonl', matrix)
        atomic_json(args.out / 'model_metadata.json', models)
        atomic_json(args.out / 'rendered_prompt_hashes.json', rendered_hashes)
        atomic_json(args.out / 'coverage_ids.json', coverage_ids(matrix))
        report.update(status='PASS_LOCAL_INPUTS', matrix_rows=len(matrix),
                      mcq_rows=sum(r['panel'] == 'mcq' for r in matrix),
                      written_rows=sum(r['panel'] == 'written' for r in matrix),
                      context_tokens=context, max_input_tokens=max(input_lengths),
                      repair_output_allowance=cap, effective_generation=effective,
                      control_prompts_tokenized=12,
                      checkpoint_generation_configs=generation_reports,
                      unused_overrides=unused, dtype='bfloat16_required_at_runtime',
                      runtime_verified=False, template_sha256=sha256_file(
                          template_dir / 'chat_template.jinja'),
                      matrix_sha256=sha256_file(args.out / 'matrix.jsonl'),
                      excluded_release_revision=release['sha'],
                      recipe_url=recipe_url, recipe_sha256=sha256_file(args.out / '7b_think_rl.sh'))
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.out / 'local-input-report.json', report)


if __name__ == '__main__':
    main()
