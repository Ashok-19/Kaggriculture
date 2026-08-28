"""Print side-by-side Phase 16.2 serial vs parallel probe timing from cloud_result.json."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def _load(output_dir: str) -> dict:
    p = Path(output_dir) / "cloud_result.json"
    if not p.exists():
        sys.exit(f"ERROR: {p} not found")
    return json.loads(p.read_text())


def _fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return str(value)
    return str(value) if value is not None else "—"


def _row(label: str, serial: object, parallel: object, speedup: bool = False) -> None:
    s = _fmt(serial)
    p = _fmt(parallel)
    if speedup and isinstance(serial, (int, float)) and isinstance(parallel, (int, float)) and parallel > 0:
        sp = f"  [{serial / parallel:.2f}x]"
    else:
        sp = ""
    print(f"  {label:<42} serial={s:<12} parallel={p}{sp}")


def _get(d: dict, *keys: str, default: object = None) -> object:
    for key in keys:
        if key in d:
            return d[key]
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True, help="serial probe output dir")
    parser.add_argument("--parallel", required=True, help="parallel probe output dir")
    args = parser.parse_args()

    s = _load(args.serial)
    p = _load(args.parallel)

    # Extract from host_pipeline section (aggregated across all attempts)
    sp = s.get("host_pipeline", {})
    pp = p.get("host_pipeline", {})

    print("\n" + "=" * 72)
    print("Phase 16.2 Lightning Probe — Serial vs Parallel Candidate Construction")
    print("=" * 72)

    print("\n[Candidate Construction]")
    _row("candidate_pool_wall_seconds",
         _get(sp, "candidate_pool_wall_seconds"), _get(pp, "candidate_pool_wall_seconds"), speedup=True)
    _row("candidate_pool_worker_busy_seconds",
         _get(sp, "candidate_pool_worker_busy_seconds"), _get(pp, "candidate_pool_worker_busy_seconds"))
    _row("candidate_pool_tasks",
         _get(sp, "candidate_pool_tasks"), _get(pp, "candidate_pool_tasks"))
    _row("candidate_workers",
         _get(sp, "candidate_workers"), _get(pp, "candidate_workers"))

    print("\n[Beam / Host]")
    for key in ("beam_search_total", "beam_search_total_wall",
                "host_pool_acquire", "motif_preparation"):
        sv = _get(sp, key)
        pv = _get(pp, key)
        if sv is not None or pv is not None:
            _row(key, sv, pv, speedup=True)

    print("\n[CUDA Evaluator]")
    for key in ("cuda_evaluator_wall", "evaluator_wall", "cuda_kernel_wall",
                "cuda_evaluator_host_wall", "cuda_kernel_seconds"):
        sv = _get(sp, key)
        pv = _get(pp, key)
        if sv is not None or pv is not None:
            _row(key, sv, pv)

    print("\n[Overall]")
    _row("cloud_wall_seconds",
         s.get("cloud_wall_seconds"), p.get("cloud_wall_seconds"), speedup=True)
    _row("cuda_games_consumed",
         s.get("cuda_games_consumed"), p.get("cuda_games_consumed"))
    _row("games_per_cloud_second",
         s.get("games_per_cloud_second"), p.get("games_per_cloud_second"))

    print("\n[Correctness]")
    _row("semantic_mismatches",
         _get(sp, "semantic_mismatches", default=s.get("semantic_mismatches", 0)),
         _get(pp, "semantic_mismatches", default=p.get("semantic_mismatches", 0)))
    _row("protected_w2l",
         _get(sp, "protected_w2l", default=s.get("protected_w2l", 0)),
         _get(pp, "protected_w2l", default=p.get("protected_w2l", 0)))
    _row("accepted_specialists",
         len(s.get("accepted_specialists", [])), len(p.get("accepted_specialists", [])))

    print("\n" + "=" * 72)
    sw = s.get("cloud_wall_seconds")
    pw = p.get("cloud_wall_seconds")
    cw_s = _get(sp, "candidate_pool_wall_seconds")
    cw_p = _get(pp, "candidate_pool_wall_seconds")
    if sw and pw:
        print(f"  Total wall speedup:               {sw / pw:.2f}x")
    if cw_s and cw_p:
        print(f"  Candidate construction speedup:   {cw_s / cw_p:.2f}x  (target >=3x)")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
