#!/usr/bin/env python3
"""Bounded read-only wait for an approved GPU; never creates a resource."""
from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from checkpoint_preflight import choose_gpu
from runpod_cli import ApiError, request

from rlvr_safety.checkpoint_study import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    state = json.loads((campaign / 'campaign.json').read_text())
    # Leave the full 90-minute initial acceptance/cleanup window, without resetting the task clock.
    last_start = datetime.fromisoformat(state['deadline']).timestamp() - 5400
    while time.time() < last_start:
        record = {'checked_at': datetime.now(UTC).isoformat(), 'checked_epoch': time.time(),
                  'resource_creation_performed': False}
        try:
            catalog = request('/catalog/gpus', {'include': 'AVAILABILITY', 'product': 'POD',
                              'cloud': 'SECURE', 'count': 1, 'minCudaVersion': '12.8'})
            atomic_json(campaign / 'capacity-watch-catalog.json', catalog)
            choice = choose_gpu(catalog)
        except (RuntimeError, ApiError) as error:
            record.update(status='WAITING', reason=str(error))
            atomic_json(campaign / 'capacity-watch.json', record)
            print(json.dumps(record), flush=True)
            if args.once:
                return 2
            time.sleep(min(60, max(0, last_start - time.time())))
        else:
            record.update(status='READY', choice=choice)
            atomic_json(campaign / 'capacity-watch.json', record)
            atomic_json(campaign / 'selected-gpu.json', choice)
            print(json.dumps(record), flush=True)
            return 0
    atomic_json(campaign / 'capacity-watch.json', {
        'status': 'DEADLINE', 'resource_creation_performed': False,
        'reason': 'No approved capacity before the last safe 90-minute canary start time.'})
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
