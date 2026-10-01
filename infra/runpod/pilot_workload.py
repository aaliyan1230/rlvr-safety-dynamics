"""Benign, pinned 7B inference probe; contains no provider credentials."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path('/workspace/rlvr-safety-dynamics/pilot')
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
USER_AGENT = 'OpenAI File Downloader, XaiImageApiFetch/1.0'


def main():
    import requests
    from huggingface_hub import configure_http_backend

    class Downloader(requests.Session):
        def send(self, request, **kwargs):
            request.headers['User-Agent'] = USER_AGENT
            return super().send(request, **kwargs)

    configure_http_backend(backend_factory=Downloader)
    ROOT.mkdir(parents=True, exist_ok=True)
    phase = {'stage': 'imports'}
    done = threading.Event()

    def progress():
        while not done.is_set():
            record = {'time': time.time(), **phase}
            with (ROOT / 'progress.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            done.wait(10)

    thread = threading.Thread(target=progress, daemon=True)
    thread.start()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        assert torch.cuda.is_available(), 'CUDA is unavailable'
        assert torch.cuda.device_count() == 1, 'Expected exactly one GPU'
        torch.manual_seed(0)
        phase['stage'] = 'download_and_load'
        started = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
            device_map={'': 0}, attn_implementation='eager',
        )
        model.eval()
        loaded = time.monotonic()
        phase['stage'] = 'inference'
        messages = [{'role': 'user', 'content': 'What is 2 + 2? Answer with just the number.'}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        torch.cuda.synchronize()
        text = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        assert text.strip() == '4', f'Unexpected benign probe output: {text!r}'
        result = {
            'passed': True, 'model': MODEL, 'revision': REVISION, 'dtype': 'bfloat16',
            'prompt': prompt, 'generation': {'max_new_tokens': 16, 'do_sample': False},
            'output': text, 'gpu': torch.cuda.get_device_name(0),
            'cuda': torch.version.cuda, 'torch': torch.__version__,
            'load_seconds': loaded - started, 'inference_seconds': time.monotonic() - loaded,
            'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
            'packages': {n: importlib.metadata.version(n) for n in (
                'transformers', 'huggingface-hub', 'accelerate', 'safetensors', 'tokenizers')},
            'repo_commit': subprocess.check_output(
                ['git', '-C', str(ROOT.parent / 'repo'), 'rev-parse', 'HEAD'], text=True).strip(),
            'pip_freeze': subprocess.check_output(
                [os.sys.executable, '-m', 'pip', 'freeze'], text=True),
        }
        serialized = json.dumps(result, indent=2) + '\n'
        (ROOT / 'result.json').write_text(serialized)
        (ROOT / 'result.sha256').write_text(hashlib.sha256(serialized.encode()).hexdigest() + '\n')
        phase['stage'] = 'complete'
        print(json.dumps({'passed': True, 'output': text, 'gpu': result['gpu']}), flush=True)
    finally:
        done.set()
        thread.join()


if __name__ == '__main__':
    main()
