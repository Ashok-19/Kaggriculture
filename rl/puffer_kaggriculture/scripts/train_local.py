#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from pufferlib import pufferl


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    return obj


def prepare_args(opponent: int, total_agents: int, minibatch: int, env_seed: int, cudagraphs: int):
    args = pufferl.load_config('kaggriculture')
    args['rank'] = 0
    args['world_size'] = 1
    args['gpu_id'] = 0
    args['nccl_id'] = b''
    args['profile'] = False
    args['cudagraphs'] = cudagraphs
    args['reset_state'] = True
    args['vec']['total_agents'] = total_agents
    args['vec']['num_buffers'] = 1
    args['vec']['num_threads'] = min(2, total_agents)
    args['env']['seed'] = env_seed
    args['env']['opponent'] = opponent
    args['train']['horizon'] = 720
    args['train']['minibatch_size'] = minibatch
    pufferl.validate_config(args)
    return args


def evaluate_checkpoint(path: Path, opponent: int, episodes: int, env_seed: int):
    args = prepare_args(opponent, episodes, 720, env_seed, -1)
    args['train']['total_timesteps'] = episodes * 720
    backend = pufferl._resolve_backend(args)
    model = backend.create_pufferl(args)
    try:
        backend.load_weights(model, str(path))
        backend.rollouts(model)
        logs = plain(dict(backend.eval_log(model)))
        util = plain(dict(backend.get_utilization(0)))
        return {'env': logs.get('env', {}), 'util': util}
    finally:
        backend.close(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=24)
    ap.add_argument('--opponent', type=int, choices=[0, 1], default=1)
    ap.add_argument('--agents', type=int, default=64)
    ap.add_argument('--minibatch', type=int, default=5760)
    ap.add_argument('--env-seed', type=int, default=424242)
    ap.add_argument('--eval-seed', type=int, default=900001)
    ap.add_argument('--eval-episodes', type=int, default=16)
    ap.add_argument('--cudagraphs', type=int, default=1)
    ap.add_argument('--out', type=Path, required=True)
    ns = ap.parse_args()
    # PufferLib load_config parses process argv again; keep only the program name
    # after consuming this harness's own arguments.
    sys.argv = [sys.argv[0]]

    ns.out.mkdir(parents=True, exist_ok=True)
    args = prepare_args(ns.opponent, ns.agents, ns.minibatch, ns.env_seed, ns.cudagraphs)
    batch = ns.agents * 720
    args['train']['total_timesteps'] = ns.epochs * batch
    backend = pufferl._resolve_backend(args)

    model = backend.create_pufferl(args)
    params = int(model.num_params())
    if params < 50_000_000:
        backend.close(model)
        raise RuntimeError(f'model below target: {params}')

    checkpoints = {
        'initial': 0,
        'early': max(1, ns.epochs // 4),
        'middle': max(1, ns.epochs // 2),
        'final': ns.epochs,
    }
    paths = {name: ns.out / f'{name}.bin' for name in checkpoints}
    backend.save_weights(model, str(paths['initial']))

    history = []
    start = time.perf_counter()
    try:
        for epoch in range(1, ns.epochs + 1):
            backend.rollouts(model)
            backend.train(model)
            log = plain(dict(backend.log(model)))
            history.append(log)
            for name, when in checkpoints.items():
                if epoch == when and name != 'initial':
                    backend.save_weights(model, str(paths[name]))
            loss = log.get('loss', {})
            env = log.get('env', {})
            util = log.get('util', {})
            print(json.dumps({
                'epoch': epoch,
                'steps': log.get('agent_steps'),
                'sps': log.get('SPS'),
                'win': env.get('win'),
                'margin': env.get('margin'),
                'policy': loss.get('policy'),
                'value': loss.get('value'),
                'entropy': loss.get('entropy'),
                'kl': loss.get('kl'),
                'clipfrac': loss.get('clipfrac'),
                'vram_gb': util.get('vram_used_gb'),
                'gpu_percent': util.get('gpu_percent'),
            }), flush=True)
    finally:
        backend.close(model)
    wall = time.perf_counter() - start

    evaluations = {}
    for name in ('initial', 'early', 'middle', 'final'):
        evaluations[name] = evaluate_checkpoint(paths[name], ns.opponent, ns.eval_episodes, ns.eval_seed)
        print('EVAL', name, json.dumps(evaluations[name], sort_keys=True), flush=True)

    finite_losses = []
    for row in history:
        for value in row.get('loss', {}).values():
            if isinstance(value, (int, float)):
                finite_losses.append(math.isfinite(value))

    result = {
        'params': params,
        'epochs': ns.epochs,
        'train_steps': ns.epochs * batch,
        'wall_seconds': wall,
        'mean_sps': (ns.epochs * batch / wall) if wall > 0 else 0,
        'config': args,
        'history': history,
        'evaluations': evaluations,
        'losses_finite': all(finite_losses) if finite_losses else False,
        'checkpoints': {k: str(v) for k, v in paths.items()},
    }
    (ns.out / 'result.json').write_text(json.dumps(result, indent=2, default=str))
    print('RESULT_PATH', ns.out / 'result.json')


if __name__ == '__main__':
    main()
