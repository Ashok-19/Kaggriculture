from __future__ import annotations

import copy
import json
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bkg4_search.module_codec import ConditionGene, ModuleGenome, RuleGene, distill_module

_B4TP_HEADER = struct.Struct("<8sIiQI")
_B4TP_ACTION = struct.Struct("<BBH")
_B4TP_UNIT = struct.Struct("<BBh")
_B4TP_ORDER = struct.Struct("<BBhi")
_B4TP_MAGIC = b"B4TP1\0\0\0"

_ITEMS = {
    "WHEAT": 0, "CARROT": 1, "TOMATO": 2, "STRAWBERRY": 3, "MELON": 4,
    "EGG": 5, "MILK": 6, "WOOL": 7, "FERTILIZER": 8,
    "GOOSE": 9, "COW": 10, "SHEEP": 11,
}
_UNIT_OPS = {
    "PASS": 0, "NORTH": 1, "SOUTH": 2, "EAST": 3, "WEST": 4,
    "PICKUP": 5, "DROP": 6, "PLACE": 7, "PLANT": 8, "WATER": 9,
    "HARVEST": 10, "FERTILIZE": 11, "DIG": 12, "BUILD_COOP": 13,
    "BUILD_PASTURE": 14, "FEED": 15, "COLLECT_FERTILIZER": 16, "CARE": 17,
}
_MARKET_OPS = {"HIRE": 1, "BUY_LAND": 2, "BUY_SEED": 3, "BUY_PRODUCT": 4, "BUY_ANIMAL": 5, "SELL": 6}


def _unit(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, list) or not value or str(value[0]) not in _UNIT_OPS:
        return 0, 0, 1
    op = _UNIT_OPS[str(value[0])]
    arg = 0
    n = 1
    if len(value) >= 2:
        if isinstance(value[1], str):
            arg = _ITEMS.get(value[1], 255)
        else:
            n = int(value[1])
    if len(value) >= 3:
        n = int(value[2])
    if not -(1 << 15) <= n < (1 << 15):
        raise ValueError("B4TP1 unit quantity must fit int16")
    return op, arg, n


def _order(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, list) or not value or str(value[0]) not in _MARKET_OPS:
        return 0, 0, 0
    op = _MARKET_OPS[str(value[0])]
    if op in (1, 2):
        return op, 0, 1
    if len(value) < 3 or not isinstance(value[1], str):
        return 0, 0, 0
    return op, _ITEMS.get(value[1], 255), int(value[2])


def write_b4tp1_from_json(source: str | Path, target: str | Path) -> None:
    obj = json.loads(Path(source).read_text(encoding="utf-8"))
    actions = obj.get("actions") if isinstance(obj, dict) else obj
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"native tape has no actions: {source}")
    meta = obj.get("source", {}) if isinstance(obj, dict) and isinstance(obj.get("source"), dict) else {}
    source_seat = int(meta.get("seat", obj.get("source_seat", -1) if isinstance(obj, dict) else -1))
    source_seed = int(meta.get("seed", obj.get("source_seed", 0) if isinstance(obj, dict) else 0))
    if source_seat not in (0, 1):
        raise ValueError("B4TP1 source seat must be 0 or 1")
    if not 0 <= source_seed < (1 << 64):
        raise ValueError("B4TP1 source seed must fit uint64")
    out = bytearray(_B4TP_HEADER.pack(_B4TP_MAGIC, 1, source_seat, source_seed, len(actions)))
    for action in actions:
        if not isinstance(action, Mapping):
            raise ValueError("native tape action must be an object")
        units = [_unit(action.get("farmer"))] + [_unit(x) for x in list(action.get("hands") or [])]
        orders = [_order(x) for x in list(action.get("market") or [])]
        if len(units) > 40 or len(orders) > 16:
            raise ValueError("native tape action exceeds engine bounds")
        out += _B4TP_ACTION.pack(len(units), len(orders), 0)
        for op, arg, n in units:
            out += _B4TP_UNIT.pack(op, arg, n)
        for op, item, n in orders:
            out += _B4TP_ORDER.pack(op, item, 0, n)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_bytes(out)


def read_b4tp1(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    if len(raw) < _B4TP_HEADER.size:
        raise ValueError("short B4TP1 tape")
    magic, version, seat, seed, count = _B4TP_HEADER.unpack_from(raw, 0)
    if magic != _B4TP_MAGIC or version != 1 or seat not in (0, 1) or not count:
        raise ValueError("invalid B4TP1 tape")
    off = _B4TP_HEADER.size
    actions = []
    for _ in range(count):
        n_units, n_orders, reserved = _B4TP_ACTION.unpack_from(raw, off); off += _B4TP_ACTION.size
        if reserved or not 1 <= n_units <= 40 or n_orders > 16:
            raise ValueError("invalid B4TP1 action header")
        units = []
        for _ in range(n_units):
            units.append(_B4TP_UNIT.unpack_from(raw, off)); off += _B4TP_UNIT.size
        orders = []
        for _ in range(n_orders):
            op, item, reserved2, n = _B4TP_ORDER.unpack_from(raw, off); off += _B4TP_ORDER.size
            if reserved2:
                raise ValueError("invalid B4TP1 order")
            orders.append((op, item, n))
        actions.append({"units": units, "market": orders})
    if off != len(raw):
        raise ValueError("trailing B4TP1 tape bytes")
    return {"source_seat": seat, "source_seed": seed, "actions": actions}


def representative_overlays(route_ids: Sequence[int]) -> tuple[ModuleGenome, ...]:
    routes = tuple(int(x) for x in route_ids)
    if len(routes) < 2:
        raise ValueError("parity requires at least two legal routes")
    r0, r1, r2 = routes[0], routes[min(1, len(routes) - 1)], routes[min(2, len(routes) - 1)]
    rows = (
        ModuleGenome(0, (RuleGene((ConditionGene("OWN_MONEY", "GE", 0, 0),), r1),)),
        ModuleGenome(37, (RuleGene((ConditionGene("DAY", "GE", 0, 1), ConditionGene("HOUR", "GE", 0, 0)), r2),)),
        ModuleGenome(0, (
            RuleGene((ConditionGene("OPP_MONEY", "GE", 0, 0),), r2),
            RuleGene((ConditionGene("OWN_MONEY", "GE", 0, 0),), r0),
        )),
    )
    for genome in rows:
        genome.validate(routes)
    return rows


def materialized_controller_source(
    parent_source: Mapping[str, Any],
    genome: ModuleGenome,
    route_ids: Sequence[int],
    *,
    route_slot: str = "slot_1",
) -> dict[str, Any]:
    genome.validate(route_ids)
    source = copy.deepcopy(dict(parent_source))
    module = distill_module(
        genome, route_ids, provenance={"materialization": "edit_search_parity"}, route_slot=str(route_slot)
    )
    root = source.get("tree")
    if not isinstance(root, dict):
        raise ValueError("parent controller source has no tree")
    if root.get("type") != "Sequence":
        root = {"type": "Sequence", "children": [root]}
        source["tree"] = root
    root.setdefault("children", []).append(module)
    return source


def _prepare_parity_cases(source_tsv: Path, work: Path, count: int) -> Path:
    lines = [x for x in source_tsv.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]
    if not lines:
        raise ValueError("parity cases TSV is empty")
    chosen = lines[: min(max(1, int(count)), len(lines))]
    cache: dict[Path, Path] = {}
    out: list[str] = []
    for line in chosen:
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError("parity case requires seven TSV fields")
        source = Path(fields[0]).resolve()
        target = cache.get(source)
        if target is None:
            target = work / f"tape_{len(cache):03d}.b4tp"
            write_b4tp1_from_json(source, target)
            cache[source] = target
        fields[0] = str(target.resolve())
        out.append("\t".join(fields))
    path = work / "cases.tsv"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def run_nonempty_parity(
    *,
    parent_dir: str | Path,
    cases_tsv: str | Path,
    cuda_evaluator: str | Path,
    work_dir: str | Path,
    case_count: int = 4,
    cpu_evaluator: str | Path | None = None,
) -> dict[str, Any]:
    import subprocess

    from tools.bkg4_search.cuda_io import pack_genomes
    from tools.edit_corpus.cuda_results import load_edit_results
    from .schema import ParentProfile

    root = Path(__file__).resolve().parents[3]
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    profile = ParentProfile.from_directory(parent_dir)
    overlays = representative_overlays(profile.route_ids)
    if not overlays or any(not g.rules for g in overlays):
        raise AssertionError("non-empty parity requires real search overlays")
    cases = _prepare_parity_cases(Path(cases_tsv).resolve(), work, int(case_count))
    genomes_path = work / "overlays.b4gm"
    pack_genomes(overlays, genomes_path)

    if cpu_evaluator is None:
        cpu_path = work / "bkg4_edit_overlay_cpp"
        source = root / "bonsai-kg" / "tests" / "bkg4_edit_overlay_cpp.cpp"
        subprocess.run(["g++", "-O3", "-std=c++17", "-o", str(cpu_path), str(source)], cwd=root, check=True)
    else:
        cpu_path = Path(cpu_evaluator).resolve()
    if not cpu_path.is_file():
        raise FileNotFoundError(f"missing CPU C++ overlay evaluator: {cpu_path}")
    cuda_path = Path(cuda_evaluator).resolve()
    if not cuda_path.is_file():
        raise FileNotFoundError(f"missing CUDA B4EC1 evaluator: {cuda_path}")

    common = [
        "--controller", str(profile.controller_path),
        "--routes", str(profile.routes_path),
        "--legacy-routes", str(profile.legacy_routes_path),
        "--cases", str(cases),
        "--genomes", str(genomes_path),
        "--feed-slot", str(profile.feed_slot),
        "--feed-count", str(profile.feed_count),
        "--strategic-route-slot", str(profile.strategic_route_slot),
    ]
    cuda_out = work / "cuda.b4ec"
    cuda_run = subprocess.run(
        [str(cuda_path), *common, "--out", str(cuda_out), "--device", "0", "--tpb", "64", "--memory-fraction", "0.50"],
        cwd=root, check=True, capture_output=True, text=True,
    )
    cuda_rows = load_edit_results(cuda_out)
    cpu_run = subprocess.run([str(cpu_path), *common], cwd=root, check=True, capture_output=True, text=True)
    cpu_rows: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for line in cpu_run.stdout.splitlines():
        if not line.strip():
            continue
        gi, ci, own, opp, fault, trace_hash = line.split("\t")
        cpu_rows[(int(gi), int(ci))] = (int(own), int(opp), int(fault), int(trace_hash))

    mismatches: list[dict[str, Any]] = []
    comparisons = 0
    cpu_faults = 0
    for gi, genome in enumerate(overlays):
        if gi >= len(cuda_rows):
            raise ValueError("CUDA result missing overlay row")
        for ci, cuda in enumerate(cuda_rows[gi]):
            comparisons += 1
            cpu = cpu_rows.get((gi, ci))
            if cpu is None:
                mismatches.append({"overlay": gi, "case": ci, "reason": "missing_cpu_row"})
                continue
            cpu_faults += int(cpu[2] != 0)
            expected = (int(cuda.own_money), int(cuda.opp_money), int(cuda.action_trace_hash))
            observed = (cpu[0], cpu[1], cpu[3])
            if cpu[2] != 0 or observed != expected:
                mismatches.append({
                    "overlay": gi,
                    "overlay_semantic_id": genome.semantic_id(),
                    "case": ci,
                    "cpu": {"own_money": cpu[0], "opp_money": cpu[1], "fault": cpu[2], "action_trace_hash": cpu[3]},
                    "cuda": {"own_money": cuda.own_money, "opp_money": cuda.opp_money, "fault": cuda.flags, "action_trace_hash": cuda.action_trace_hash},
                })
    report = {
        "parent_identity": profile.combined_agent_identity,
        "controller_semantic_hash": profile.controller_semantic_hash,
        "overlay_semantic_ids": [g.semantic_id() for g in overlays],
        "nonempty_overlays": len(overlays),
        "cases": len(cuda_rows[0]) if cuda_rows else 0,
        "comparisons": comparisons,
        "cpu_faults": cpu_faults,
        "cuda_faults": 0,
        "cpu_cuda_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
        "cuda_stdout": cuda_run.stdout.strip(),
    }
    return report


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Bounded non-empty ModuleGenome CPU C++ ↔ CUDA B4EC1 parity")
    ap.add_argument("--parent-dir", required=True)
    ap.add_argument("--cases-tsv", required=True)
    ap.add_argument("--cuda-evaluator", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--case-count", type=int, default=4)
    ap.add_argument("--cpu-evaluator")
    ap.add_argument("--report")
    args = ap.parse_args()
    report = run_nonempty_parity(
        parent_dir=args.parent_dir,
        cases_tsv=args.cases_tsv,
        cuda_evaluator=args.cuda_evaluator,
        work_dir=args.work_dir,
        case_count=args.case_count,
        cpu_evaluator=args.cpu_evaluator,
    )
    text = json.dumps(report, sort_keys=True, indent=2)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if report["cpu_cuda_mismatches"] or report["cpu_faults"] or report["cuda_faults"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
