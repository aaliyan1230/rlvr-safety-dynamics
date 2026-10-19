"""Legacy instrument adapter over the already tested pinned BF16 provider."""
from __future__ import annotations

from pathlib import Path

from .checkpoint_study import digest
from .permission.generation import HFPermissionProvider


class HFCheckpointProvider:
    def __init__(self, config: dict, metadata: dict, cache: Path):
        self.provider = HFPermissionProvider(config, metadata, cache)
        self.revision = config['model']['revision']
        self.config = config
        import torch
        if any(p.dtype != torch.bfloat16 for p in self.provider.model.parameters()
               if p.is_floating_point()):
            self.provider.close()
            raise ValueError('actual parameter dtype differs from BF16')

    def __call__(self, question: dict, protocol_hash: str) -> dict:
        if question['revision'] != self.revision:
            raise ValueError('loaded checkpoint differs from requested row')
        messages = [{'role': 'system', 'content': question['system_prompt']},
                    {'role': 'user', 'content': question['prompt']}]
        request = {'messages': messages, 'condition_id': question['id'], 'sample_id': '0'}
        response = self.provider.respond(request)
        ids = response['generated_token_ids']
        # Keep reasoning markers even if a tokenizer classifies them as special.
        text_ids = ids[:-1] if ids and ids[-1] in self.provider.eos_ids else ids
        raw = self.provider.tokenizer.decode(text_ids, skip_special_tokens=False)
        rendered = response['rendered_prompt']
        return {**response, 'id': question['id'], 'input_hash': question['input_hash'],
                'protocol_hash': protocol_hash, 'raw_response': raw,
                'model_revision_resolved': self.revision, 'stop': response['stop_status'],
                'reasoning_open': rendered.count('<think>') > rendered.count('</think>'),
                'runtime_hash': digest(self.provider.provenance)}

    def close(self):
        self.provider.close()
