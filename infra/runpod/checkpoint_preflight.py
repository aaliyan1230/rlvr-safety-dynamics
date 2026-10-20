"""Read-only native preflight and conservative exposure checks for OLMo batches."""
from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

from runpod_cli import list_pods, request

from rlvr_safety.checkpoint_bundle import verify_prepared_bundle
from rlvr_safety.checkpoint_study import digest, enforce_exposure
from rlvr_safety.io import sha256_file

GPU_CHOICES = (('NVIDIA H100 80GB HBM3', 4.50), ('NVIDIA A100-SXM4-80GB', 2.00))


class CapacityUnavailable(ValueError):
    """Read-only preflight found no slot; no create request has occurred."""


def campaign_state_paths(campaign: Path) -> list[Path]:
    """A predeclared fallback shares its parent's clock, billing baseline and caps."""
    campaign = campaign.resolve()
    limits = json.loads((campaign / 'campaign.json').read_text())
    roots = [campaign]
    for name in limits.get('prior_campaigns', []):
        prior = (campaign / name).resolve()
        if prior in roots:
            raise ValueError('duplicate or cyclic campaign budget roots')
        previous = json.loads((prior / 'campaign.json').read_text())
        shared = ('started_at', 'deadline', 'gpu_ceiling_usd', 'api_ceiling_usd')
        if previous.get('prior_campaigns') or any(
                previous.get(key) != limits.get(key) for key in shared):
            raise ValueError('fallback must retain the original campaign clock and caps')
        if sha256_file(prior / 'billing-before.json') != sha256_file(
                campaign / 'billing-before.json'):
            raise ValueError('fallback must retain the original native billing baseline')
        roots.append(prior)
    return sorted(path for root in roots for path in root.glob('cloud/*/state.json'))


def readiness_fingerprint(protocol: dict) -> str:
    return digest({k: v for k, v in protocol.items() if k != 'generation'} | {
        'generation': {k: v for k, v in protocol['generation'].items() if k != 'max_new_tokens'}})


def choose_gpu(catalog: dict) -> dict:
    for gpu_id, ceiling in GPU_CHOICES:
        candidates = [g for g in catalog['gpus'] if g['id'] == gpu_id]
        if not candidates:
            continue
        gpu = candidates[0]
        if not 0 < gpu['price']['secure'] <= ceiling:
            continue
        options = [d for d in gpu.get('dataCenters', []) if d['availability'] != 'NONE']
        if options:
            selected = max(options, key=lambda d: {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}.get(
                d['availability'], 0))
            return {'gpu': gpu_id, 'hourly_ceiling': ceiling,
                    'catalog_price': gpu['price']['secure'],
                    'datacenter': selected['id'], 'availability': selected['availability']}
    raise RuntimeError('neither allowed GPU type has native catalog capacity within its ceiling')


def native_preflight(bundle: Path, campaign: Path, *, balance: dict,
                     launch: dict) -> dict:
    verify_prepared_bundle(bundle)
    state = json.loads((campaign / 'campaign.json').read_text())
    protocol = json.loads((bundle / 'protocol.json').read_text())
    batch = json.loads((bundle / 'batch.json').read_text())
    accepted = json.loads((campaign / 'unpaid-readiness.json').read_text())
    if (accepted.get('stage_a') != 'PASS' or accepted.get('stage_b') != 'PASS'
            or accepted.get('protocol_fingerprint') != readiness_fingerprint(protocol)):
        raise ValueError('complete unpaid readiness evidence does not match this prepared protocol')
    for name, checksum in accepted['evidence'].items():
        if sha256_file(campaign / name) != checksum:
            raise ValueError('accepted readiness evidence changed')
    now = time.time()
    if (balance.get('source') not in ('RunPod native billing page', 'RunPod native console balance')
            or not math.isfinite(balance['usd']) or not math.isfinite(balance['observed_epoch'])
            or not -60 <= now - balance['observed_epoch'] <= 86400 or balance['usd'] <= 0):
        raise ValueError('need a recent native credit balance; estimates are insufficient')
    if launch['gpu'] != protocol['gpu'] or launch['image'] != protocol['image']:
        raise ValueError('launch differs from the prepared protocol')
    if launch['stage'] != batch['stage']:
        raise ValueError('launch stage differs from the prepared batch')
    if not 900 <= launch['max_seconds'] <= 21600 or launch['disk_gb'] != 80:
        raise ValueError('unapproved lifetime or disk size')
    catalog = request('/catalog/gpus', {'include': 'AVAILABILITY', 'product': 'POD',
                      'cloud': 'SECURE', 'count': 1, 'minCudaVersion': '12.8'})
    gpu = next(g for g in catalog['gpus'] if g['id'] == launch['gpu'])
    allowed = dict(GPU_CHOICES)
    if launch['hourly_ceiling'] != allowed.get(launch['gpu']):
        raise ValueError('GPU/rate is outside the agreed hardware choices')
    if not 0 < gpu['price']['secure'] <= launch['hourly_ceiling']:
        raise ValueError('catalog quote exceeds the approved ceiling')
    if launch['datacenter'] == 'auto':
        options = [d for d in gpu.get('dataCenters', []) if d['availability'] != 'NONE']
        if not options:
            raise CapacityUnavailable('selected GPU has no current native capacity')
        best = max(options, key=lambda d: {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}.get(
            d['availability'], 0))
        launch = {**launch, 'datacenter': best['id']}
    if not any(d['id'] == launch['datacenter'] and d['availability'] != 'NONE'
               for d in gpu.get('dataCenters', [])):
        raise CapacityUnavailable('selected GPU has no native capacity in the selected data center')
    pods, volumes = list_pods(), request('/network-volumes')['networkVolumes']
    if pods or volumes:
        raise ValueError('unexpected account resources: inspect before launching')
    billing = request('/billing', {'startTime': '2026-10-01T00:00:00Z',
                      'endTime': datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ'),
                      'bucketSize': 'day'})
    baseline = json.loads((campaign / 'billing-before.json').read_text())
    native_delta = billing['metadata']['totals']['totalAmount'] - baseline['metadata']['totals'][
        'totalAmount']
    if native_delta < -.001:
        raise ValueError('native usage decreased; reconcile account adjustments before launch')
    projections, readiness_projections, canary_projections = [], [], []
    for path in campaign_state_paths(campaign):
        prior = json.loads(path.read_text())
        if not prior.get('cleanup_verified'):
            raise ValueError('previous batch has unverified cleanup')
        elapsed = prior['finished_at'] - prior['started_at']
        maximum = projected_exposure(prior['launch'], elapsed)
        projections.append(maximum)
        if prior['launch']['stage'] != 'main':
            readiness_projections.append(maximum)
        if prior['launch']['stage'] == 'canary':
            canary_projections.append(maximum)
    accounted = max(native_delta, sum(projections))
    maximum = projected_exposure(launch, launch['max_seconds'])
    deadline = datetime.fromisoformat(state['deadline']).timestamp()
    remaining_gpu = min(300, state['gpu_ceiling_usd']) - accounted
    enforce_exposure(deadline=deadline, remaining_gpu=remaining_gpu,
                     remaining_api=state['api_ceiling_usd'], maximum_gpu=maximum,
                     maximum_api=0, seconds=launch['max_seconds'], now=now)
    if batch['stage'] != 'main' and sum(readiness_projections) + maximum > 60:
        raise ValueError('readiness GPU exposure would exceed $60')
    if batch['stage'] == 'canary' and sum(canary_projections) + maximum > 10:
        raise ValueError('initial canary GPU exposure would exceed $10')
    if balance['usd'] < maximum * 2:
        raise ValueError('native balance is too low for the bounded launch and cleanup buffer')
    return {'ok': True, 'checked_epoch': now, 'launch': launch, 'native_billing': billing,
            'native_usage_delta_usd': native_delta,
            'conservative_elapsed_projection_usd': accounted,
            'maximum_launch_exposure_usd': maximum, 'balance': balance,
            'inventory_empty': True, 'stage': batch['stage'],
            'readiness_fingerprint': readiness_fingerprint(protocol),
            'bundle_sha256': sha256_file(bundle.with_suffix('.tar.gz')),
            'projection_note': 'Catalog ceiling plus conservative $0.20/GB/month storage '
            'allowance and 25% fee/tax buffer; projections are not reported actual charges.'}


def projected_exposure(launch: dict, seconds: float) -> float:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('unknown or negative billed runtime cannot mean zero cost')
    return (launch['hourly_ceiling'] + launch['disk_gb'] * .20 / 720) * seconds / 3600 * 1.25
