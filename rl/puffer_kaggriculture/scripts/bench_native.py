#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch
from pufferlib import pufferl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--updates', type=int, default=4)
    ap.add_argument('--agents', type=int, default=64)
    ap.add_argument('--minibatch', type=int, default=5760)
    ap.add_argument('--cudagraphs', type=int, default=1)
    ns = ap.parse_args()
    # PufferLib load_config parses process argv again; keep only the program name
    # after consuming this harness's own arguments.
    sys.argv = [sys.argv[0]]

    args = pufferl.load_config('kaggriculture')
    args.update(rank=0, world_size=1, gpu_id=0, nccl_id=b'', profile=False,
                cudagraphs=ns.cudagraphs, reset_state=True)
    args['vec']['total_agents'] = ns.agents
    args['vec']['num_buffers'] = 1
    args['vec']['num_threads'] = min(2, ns.agents)
    args['train']['horizon'] = 720
    args['train']['minibatch_size'] = ns.minibatch
    args['train']['total_timesteps'] = ns.updates * ns.agents * 720
    pufferl.validate_config(args)

    backend = pufferl._resolve_backend(args)
    create_start = time.perf_counter()
    model = backend.create_pufferl(args)
    create_seconds = time.perf_counter() - create_start
    params = int(model.num_params())
    if params < 50_000_000:
        backend.close(model)
        raise RuntimeError(f'model below 50M target: {params}')

    rows = []
    try:
        for update in range(ns.updates):
            torch.cuda.synchronize()
            start = time.perf_counter()
            backend.rollouts(model)
            torch.cuda.synchronize()
            rollout_seconds = time.perf_counter() - start

            start = time.perf_counter()
            backend.train(model)
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - start

            log = dict(backend.log(model))
            loss = log.get('loss', {})
            finite = [math.isfinite(v) for v in loss.values() if isinstance(v, (int, float))]
            util = dict(backend.get_utilization(0))
            steps = ns.agents * 720
            row = {
                'update': update,
                'sps': steps / (rollout_seconds + train_seconds),
                'rollout_sps': steps / rollout_seconds,
                'rollout_seconds': rollout_seconds,
                'train_seconds': train_seconds,
                'losses_finite': bool(finite) and all(finite),
                'vram_used_gb': util.get('vram_used_gb'),
                'gpu_percent': util.get('gpu_percent'),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        backend.close(model)

    steady = rows[1:] if len(rows) > 1 else rows
    print(json.dumps({
        'params': params,
        'create_seconds': create_seconds,
        'steady_mean_sps': sum(r['sps'] for r in steady) / len(steady),
        'steady_min_sps': min(r['sps'] for r in steady),
        'steady_max_sps': max(r['sps'] for r in steady),
        'all_losses_finite': all(r['losses_finite'] for r in rows),
    }), flush=True)


if __name__ == '__main__':
    main()
