from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SEARCH = ROOT / "tools" / "bkg4_search"
BKG_PY = ROOT / "bonsai-kg" / "python"
for p in (SEARCH, BKG_PY):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import bonsai_kg_v4 as bkg  # noqa: E402
from information_state_io import load_r0, registry_manifest  # noqa: E402
from tools.bkg4_search.invariant_schema import invariant_model_input, module_from_model_input  # noqa: E402,F401

from model import CASE_DIM, EDIT_CAT_COLS, EDIT_NUM_COLS, EditFormerConfig, TREE_CAT_COLS, TREE_NUM_COLS

ROUTES = (22, 23, 29, 32, 34, 46, 63, 64, 70, 77, 79, 211, 212, 213, 214)
ROUTE_POS = {x: i + 1 for i, x in enumerate(ROUTES)}
LEGAL_FEATURES = (
    "STEP", "DAY", "HOUR", "SHOP_AT", "SHOP_SEEN", "OWN_MONEY", "OPP_MONEY", "OWN_UNITS", "OPP_UNITS",
    "OWN_LAND", "OPP_LAND", "OWN_HIRES_TODAY", "OPP_HIRES_TODAY", "MARKET_PRICE", "MARKET_INVENTORY",
    "OWN_SHED", "OWN_SEED", "OWN_WORKER_X", "OWN_WORKER_Y", "OPP_WORKER_X", "OPP_WORKER_Y",
    "OWN_TILE_KIND", "OPP_TILE_KIND", "OWN_TILE_ITEM", "OPP_TILE_ITEM", "OWN_TILE_YIELD", "OPP_TILE_YIELD",
)
FEATURE_POS = {bkg.FEATURES[n]: i + 1 for i, n in enumerate(LEGAL_FEATURES)}
FEATURE_NAME = {bkg.FEATURES[n]: n for n in LEGAL_FEATURES}
CMP_NAMES = ("EQ", "NE", "LE", "GE")
CMP_CODES = tuple(bkg.CMP[x] for x in CMP_NAMES)
CMP_POS = {x: i + 1 for i, x in enumerate(CMP_CODES)}
CMP_NAME = {bkg.CMP[x]: x for x in CMP_NAMES}

FH = struct.Struct("<8sII")
GH = struct.Struct("<HBB")
RH = struct.Struct("<BBH")
COND = struct.Struct("<HBBhHi")
RESH = struct.Struct("<8sIII")
RES = struct.Struct("<iiHH")
GENOME_SIZE = 420

HIST_CAMPAIGNS = (
    "campaign_broad_evo4096", "campaign_guarded_evo4096", "campaign_p144_broad_evo4096",
    "campaign_evo2048_v2", "campaign_greedy2048_v2",
)
OUT = ROOT / "scratch" / "bkg4_editformer_jax"
MIN_GRU_OUT = ROOT / "scratch" / "bkg4_rl_synthesis_aug20"
HIST40_CASES = ROOT / "scratch" / "bkg4_search" / "cuda_combined40.tsv"
CURRENT_CASES = ROOT / "scratch" / "bkg4_search" / "current_shifted_aug20" / "cases.tsv"
HIST_R0 = OUT / "historical40_r0.b4is"
CURRENT_R0 = OUT / "current140_r0_264.b4is"
BASE0_IDENTITY = ROOT / "bonsai-kg" / "integrations" / "bkg4_champion" / "assets" / "identity.json"
CURRENT_IDENTITY = ROOT / "bonsai-kg" / "integrations" / "bkg4_champion" / "champions" / "w33_s172_expansion_v1" / "identity.json"
BASE0_CASES_JSONL = ROOT / "scratch" / "bkg4_search" / "base0_discovery16_p72" / "cases.jsonl"
EXPECTED_BASE0_COMBINED = "601f4d0914564e07485b08dc6f563eb497a93316391225fdce1cbaf5fa59e4ea"
EXPECTED_CURRENT_COMBINED = "6359ab400f9d128945cdbba5df7178a33b88c643ddd9b488299d66e14daa6925"
EXPECTED_CURRENT_GENOMES_SHA = "a22790237f2ef819be67ca6b9fc31abf795ee87e20190bcf9e554bcdc0b1f987"
EXPECTED_CURRENT_RESULTS_SHA = "3a78dfbd2a6c602ede774f4681ffebcce34b2d6b51a96a2f120fe07090edeb32"
EXPECTED_CURRENT_CASES_SHA = "5fc22a4272ecc71e08e8f561547bc3f5cf362480e864512f897eae8c84579fa5"


@dataclass
class DecodedGenome:
    prefix: int
    rules: list[dict]


@dataclass
class Row:
    nc: int
    blob: bytes
    baseline: np.ndarray
    margins: np.ndarray
    campaign: str
    blob_sha: str
    semantic_id: str
    labels: dict
    meta: dict
    parent_kind: int  # 1=BASE_0, 2=current345


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(parts: Iterable[str | bytes]) -> str:
    h = hashlib.sha256()
    for part in parts:
        z = part.encode() if isinstance(part, str) else part
        h.update(struct.pack("<I", len(z)))
        h.update(z)
    return h.hexdigest()


def signed_log(v: int | float, scale: float = 200000.0) -> float:
    v = float(v)
    return math.copysign(math.log1p(abs(v)) / math.log1p(scale), v) if v else 0.0


def decode_genome(blob: bytes) -> DecodedGenome:
    if len(blob) != GENOME_SIZE:
        raise ValueError(f"bad genome size {len(blob)}")
    prefix, nr, _ = GH.unpack_from(blob, 0)
    pos = GH.size
    rules = []
    for ri in range(8):
        ncond, _, route = RH.unpack_from(blob, pos)
        pos += RH.size
        conds = []
        for _ in range(4):
            fid, cmp_, _, idx, _, val = COND.unpack_from(blob, pos)
            pos += COND.size
            conds.append((int(fid), int(cmp_), int(idx), int(val)))
        if ri < nr:
            rules.append({"route": int(route), "conds": conds[: int(ncond)]})
    return DecodedGenome(int(prefix), rules)


def label_dict(baseline: np.ndarray, margins: np.ndarray) -> dict:
    if baseline.shape != margins.shape:
        raise ValueError("baseline/candidate margin mismatch")
    win = baseline > 0
    loss = baseline < 0
    l2w = int(np.sum(loss & (margins > 0)))
    w2l = int(np.sum(win & (margins <= 0)))
    delta = margins - baseline
    reg3 = int(np.sum(win & (delta < -3000)))
    md = float(np.mean(delta))
    worst = float(np.min(delta[win])) if np.any(win) else 0.0
    safe = float(l2w > 0 and w2l == 0 and reg3 == 0)
    q = math.tanh((6 * l2w - 12 * w2l - 4 * reg3 + md / 2000.0) / 10.0)
    return {
        "safe": safe,
        "flip": float(l2w > 0),
        "regression": float(w2l > 0),
        "l2w": l2w,
        "w2l": w2l,
        "reg3": reg3,
        "mean_delta": md,
        "margin_target": math.tanh(md / 5000.0),
        "worst_win_delta": worst,
        "worst_target": math.tanh(worst / 5000.0),
        "rank_target": q,
    }


def _read_result(path: Path, expected_genomes: int, expected_cases: int) -> np.ndarray:
    raw = path.read_bytes()
    magic, version, ng, nc = RESH.unpack_from(raw, 0)
    if magic != b"B4RS1\0\0\0" or version != 1:
        raise ValueError(f"invalid result header {path}")
    if ng != expected_genomes or nc != expected_cases:
        raise ValueError(f"probe/case alignment mismatch {path}: got {ng}x{nc}, expected {expected_genomes}x{expected_cases}")
    expected = RESH.size + ng * nc * RES.size
    if len(raw) != expected:
        raise ValueError(f"result payload mismatch {path}")
    out = np.empty((ng, nc), np.int32)
    for gi in range(ng):
        for ci in range(nc):
            own, opp, _, flags = RES.unpack_from(raw, RESH.size + (gi * nc + ci) * RES.size)
            if flags:
                raise ValueError(f"candidate result flag {flags} at {path} genome={gi} case={ci}")
            out[gi, ci] = own - opp
    return out


def _historical_baseline_truth() -> np.ndarray:
    vals=[]
    identities=set()
    for line in HIST40_CASES.read_text().splitlines():
        if not line.strip():
            continue
        fields=line.split("\t")
        vals.append(int(round(float(fields[-1]))))
        base_tape=Path(fields[0])
        record=json.loads(base_tape.read_text())
        identity=(record.get("source") or {}).get("base0_identity")
        if identity is None:
            raise ValueError(f"historical base tape missing source.base0_identity: {base_tape}")
        identities.add(str(identity))
    if len(vals)!=40: raise ValueError("historical combined40 benchmark changed")
    if identities!={EXPECTED_BASE0_COMBINED}:
        raise ValueError(f"historical BASE_0 identity mismatch: {identities}")
    return np.asarray(vals,np.int32)


def history_rows() -> tuple[list[Row], int]:
    best: dict[str, Row] = {}
    raw_count = 0
    truth = _historical_baseline_truth()
    for dn in HIST_CAMPAIGNS:
        d = ROOT / "scratch" / "bkg4_search" / dn
        sr = json.loads((d / "search_result.json").read_text())
        baseline = np.asarray(sr["baseline"], np.int32)
        if not np.array_equal(baseline,truth[:len(baseline)]): raise ValueError(f"historical benchmark/order mismatch {dn}")
        if len(baseline) not in (16, 40):
            raise ValueError(f"unexpected historical case count {dn}: {len(baseline)}")
        for gm in sorted(d.glob("batch_*.b4gm")):
            rs = gm.with_suffix(".b4rs")
            if not rs.exists():
                continue
            raw = gm.read_bytes()
            magic, version, ng = FH.unpack_from(raw, 0)
            if magic != b"B4GM1\0\0\0" or version != 1:
                raise ValueError(f"invalid genome file {gm}")
            results = _read_result(rs, ng, len(baseline))
            for gi in range(ng):
                raw_count += 1
                blob = raw[FH.size + gi * GENOME_SIZE: FH.size + (gi + 1) * GENOME_SIZE]
                sid = hashlib.sha256(blob).hexdigest()
                rec = Row(len(baseline), blob, baseline.copy(), results[gi].copy(), dn, sid, sid,
                          label_dict(baseline, results[gi]), {}, 1)
                if sid not in best or rec.nc > best[sid].nc:
                    best[sid] = rec
    rows = list(best.values())
    return rows, raw_count


def current_rows() -> list[Row]:
    gm_path = MIN_GRU_OUT / "single_predicates.b4gm"
    rs_path = MIN_GRU_OUT / "single_predicates_current140.b4rs"
    if sha256_file(gm_path)!=EXPECTED_CURRENT_GENOMES_SHA or sha256_file(rs_path)!=EXPECTED_CURRENT_RESULTS_SHA or sha256_file(CURRENT_CASES)!=EXPECTED_CURRENT_CASES_SHA:
        raise ValueError("frozen current400 benchmark identity changed")
    parent_spec(2)
    raw = gm_path.read_bytes()
    magic, version, ng = FH.unpack_from(raw, 0)
    if magic != b"B4GM1\0\0\0" or version != 1:
        raise ValueError("invalid current genome file")
    results = _read_result(rs_path, ng, 140)
    baseline = results[0].copy()
    if (int(np.sum(baseline > 0)), int(np.sum(baseline < 0))) != (98, 42):
        raise ValueError("current parent no longer matches frozen 98W/42L synthesis benchmark")
    meta = [json.loads(x) for x in (MIN_GRU_OUT / "single_predicates_meta.jsonl").read_text().splitlines() if x.strip()]
    if len(meta) != ng:
        raise ValueError("current meta/genome index mismatch")
    rows = []
    for gi in range(1, ng):
        blob = raw[FH.size + gi * GENOME_SIZE: FH.size + (gi + 1) * GENOME_SIZE]
        sid = hashlib.sha256(blob).hexdigest()
        semantic = str(meta[gi].get("semantic_id") or sid)
        rows.append(Row(140, blob, baseline.copy(), results[gi].copy(), "current_single", sid, semantic,
                        label_dict(baseline, results[gi]), dict(meta[gi]), 2))
    # deterministic heuristic rank percentile; high score = good.
    scores = np.asarray([float(r.meta.get("heuristic_score", 0.0)) for r in rows])
    order = np.argsort(-scores, kind="mergesort")
    rank = np.empty(len(rows), np.int32); rank[order] = np.arange(len(rows))
    for i, r in enumerate(rows):
        r.meta["heuristic_rank"] = int(rank[i])
        r.meta["heuristic_norm"] = float(np.tanh(scores[i] / 50.0))
    return rows


def stratified_split(rows: list[Row], frac: float = 0.2) -> tuple[list[int], list[int]]:
    groups = {0: [], 1: []}
    for i, r in enumerate(rows):
        groups[int(r.labels["safe"])].append((r.blob_sha, i))
    train, val = [], []
    for items in groups.values():
        items.sort()
        every = max(2, round(1 / frac))
        for j, (_, i) in enumerate(items):
            (val if j % every == 0 else train).append(i)
    return train, val


def grouped_secondary_split(rows: list[Row], frac: float = 0.2) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        sk = threshold_skeleton(r.blob)
        groups.setdefault(sk, []).append(i)
    train, val = [], []
    for key, ids in sorted(groups.items()):
        bucket = int(key[:8], 16) % max(2, round(1 / frac))
        (val if bucket == 0 else train).extend(ids)
    return train, val


def threshold_skeleton(blob: bytes) -> str:
    z = bytearray(blob)
    prefix, nr, _ = GH.unpack_from(z, 0)
    pos = GH.size
    for ri in range(8):
        ncond, _, _ = RH.unpack_from(z, pos); pos += RH.size
        for ci in range(4):
            if ri < nr and ci < ncond:
                # zero only condition scalar value; preserve feature/cmp/index/topology.
                val_off = pos + 8
                z[val_off:val_off + 4] = b"\0\0\0\0"
            pos += COND.size
    return hashlib.sha256(z).hexdigest()


def leakage_audit(rows: list[Row], train: list[int], val: list[int]) -> dict:
    ts, vs = set(train), set(val)
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(threshold_skeleton(r.blob), []).append(i)
    crossing = [ids for ids in groups.values() if any(i in ts for i in ids) and any(i in vs for i in ids)]
    return {
        "skeleton_groups": len(groups),
        "cross_split_near_duplicate_groups": len(crossing),
        "rows_in_crossing_groups": sum(len(x) for x in crossing),
    }


def _case_meta_historical(n: int) -> list[dict]:
    rows = []
    for line in HIST40_CASES.read_text().splitlines()[:n]:
        z = line.split("\t")
        rows.append({"seat": int(z[2]), "seed": int(z[3]), "family": int(z[5]), "protected": int(z[6]), "own_history": int(z[7])})
    return rows


def _case_meta_current() -> list[dict]:
    rows = []
    for line in CURRENT_CASES.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        z = line.split("\t")
        if len(z) != 6:
            raise ValueError("current cases manifest changed")
        rows.append({"seat": int(z[1]), "seed": int(z[2]), "family": int(z[3]), "protected": int(z[4]), "own_history": int(z[5])})
    if len(rows) != 140:
        raise ValueError("current140 manifest changed")
    return rows


def _registry_lookup() -> dict[tuple[int, int], int]:
    out = {}
    for off, (name, args) in enumerate(registry_manifest()):
        if len(args) <= 1:
            out[(bkg.FEATURES[name], args[0] if args else 0)] = off
    return out


def capture_matrix(path: Path):
    r = load_r0(path)
    arr = np.asarray(r.data, dtype=np.int32).reshape(r.cases, len(r.probes), r.width)
    return r, arr


def _cmp(vals: np.ndarray, code: int, value: int) -> np.ndarray:
    if code == bkg.CMP["EQ"]: return vals == value
    if code == bkg.CMP["NE"]: return vals != value
    if code == bkg.CMP["LE"]: return vals <= value
    if code == bkg.CMP["GE"]: return vals >= value
    raise ValueError(f"unsupported comparator {code}")


def _slack(vals: np.ndarray, code: int, value: int) -> np.ndarray:
    denom = 1.0 + abs(float(value))
    if code == bkg.CMP["EQ"]: return -np.abs(vals - value) / denom
    if code == bkg.CMP["NE"]: return np.where(vals != value, np.abs(vals - value) / denom, -1.0)
    if code == bkg.CMP["LE"]: return (value - vals) / denom
    if code == bkg.CMP["GE"]: return (vals - value) / denom
    return np.full(vals.shape, -1.0, np.float32)


def parent_spec(kind: int) -> dict:
    path = BASE0_IDENTITY if kind == 1 else CURRENT_IDENTITY
    ident = json.loads(path.read_text()); combined=ident["combined_agent_identity"]
    expected=EXPECTED_BASE0_COMBINED if kind==1 else EXPECTED_CURRENT_COMBINED
    if combined!=expected: raise ValueError(f"frozen parent identity changed: {combined} != {expected}")
    if kind == 1:
        return {"kind": 1, "nodes": 99, "bb": 4, "state": 0, "strategic_slot": -1,
                "combined": combined, "semantic": ident["controller_semantic_hash"]}
    return {"kind": 2, "nodes": 345, "bb": 23, "state": 0, "strategic_slot": 19,
            "combined": combined, "semantic": ident["controller_semantic_hash"]}


def build_tree(decoded: DecodedGenome, kind: int, cfg: EditFormerConfig):
    tc = np.zeros((cfg.tree_cap, TREE_CAT_COLS), np.int16)
    tn = np.zeros((cfg.tree_cap, TREE_NUM_COLS), np.float32)
    mask = np.zeros(cfg.tree_cap, bool)
    parent = np.full(cfg.tree_cap, -1, np.int16)
    depth = np.zeros(cfg.tree_cap, np.int8)
    rule_id = np.zeros(cfg.tree_cap, np.int8)
    ps = parent_spec(kind)

    def put(i, typ, feat=0, cmp_=0, route=0, ordinal=0, par=-1, dep=0, rid=0, nums=None):
        if i >= cfg.tree_cap: return
        tc[i] = (typ, feat, cmp_, route, kind, ordinal)
        if nums is not None: tn[i, :len(nums)] = nums
        mask[i] = True; parent[i] = par; depth[i] = dep; rule_id[i] = rid

    put(0, 1, ordinal=0, nums=[1, 0, 0, decoded.prefix/719, len(decoded.rules)/8, 0, ps["nodes"]/512, ps["bb"]/32, max(ps["strategic_slot"],0)/32, 0])
    put(1, 2, ordinal=1, par=0, dep=1, nums=[1,0,0,0,0,0,ps["nodes"]/512,ps["bb"]/32,max(ps["strategic_slot"],0)/32,0])
    put(2, 3, route=ROUTE_POS.get(213,0), ordinal=2, par=1, dep=2, nums=[1,0,0,0,0,0,ps["nodes"]/512,ps["bb"]/32,max(ps["strategic_slot"],0)/32,0])
    pos = 3
    for ri, rr in enumerate(decoded.rules, 1):
        if pos >= cfg.tree_cap: break
        rp = ROUTE_POS.get(rr["route"], 0)
        rpos = pos
        put(rpos, 4, route=rp, ordinal=min(ri,11), par=0, dep=1, rid=ri,
            nums=[1,0,0,decoded.prefix/719,len(decoded.rules)/8,len(rr["conds"])/4,ps["nodes"]/512,ps["bb"]/32,max(ps["strategic_slot"],0)/32,0])
        pos += 1
        for ci, (fid, cmp_, idx, val) in enumerate(rr["conds"]):
            if pos >= cfg.tree_cap: break
            put(pos, 5, FEATURE_POS.get(fid,0), CMP_POS.get(cmp_,0), rp, min(ci+1,11), rpos, 2, ri,
                [1,signed_log(val),max(-1,min(1,idx/100)),decoded.prefix/719,len(decoded.rules)/8,len(rr["conds"])/4,
                 ps["nodes"]/512,ps["bb"]/32,max(ps["strategic_slot"],0)/32,ci/3 if ci else 0])
            pos += 1
    return tc, tn, mask, parent, depth, rule_id


def _window_for_rule(rr: dict, prefix: int):
    start, end = prefix, 719
    semantic = None
    for c in rr["conds"]:
        fid, cmp_, idx, val = c
        if FEATURE_NAME.get(fid) == "STEP":
            if cmp_ == bkg.CMP["GE"]: start = max(start, val)
            elif cmp_ == bkg.CMP["LE"]: end = min(end, val)
            elif cmp_ == bkg.CMP["EQ"]: start = end = val
        elif semantic is None:
            semantic = c
    return start, end, semantic


def build_edit(decoded: DecodedGenome, row: Row, cfg: EditFormerConfig):
    ec = np.zeros((cfg.edit_cap, EDIT_CAT_COLS), np.int16)
    en = np.zeros((cfg.edit_cap, EDIT_NUM_COLS), np.float32)
    mask = np.zeros(cfg.edit_cap, bool)
    heuristic = float(row.meta.get("heuristic_norm", 0.0))
    hrank = float(row.meta.get("heuristic_rank", 0)) / max(1, 399)
    subtree = sum(1 + len(r["conds"]) for r in decoded.rules)
    ec[0] = (1,0,0,0,0); mask[0] = True
    en[0] = [1,decoded.prefix/719,len(decoded.rules)/8,0,0,0,0,0,decoded.prefix/719,1,1-decoded.prefix/719,heuristic,hrank,subtree/40]
    pos = 1
    for ri, rr in enumerate(decoded.rules,1):
        if pos >= cfg.edit_cap-1: break
        start,end,sem = _window_for_rule(rr, decoded.prefix)
        feat=cmp_=idx=val=0
        if sem is not None: feat,cmp_,idx,val=sem
        ec[pos] = (2, FEATURE_POS.get(feat,0), CMP_POS.get(cmp_,0), ROUTE_POS.get(rr["route"],0), min(ri,11))
        en[pos] = [1,decoded.prefix/719,len(decoded.rules)/8,len(rr["conds"])/4,0,signed_log(val),signed_log(val),max(-1,min(1,idx/100)),start/719,end/719,max(0,end-start+1)/720,heuristic,hrank,(1+len(rr["conds"]))/40]
        mask[pos]=True; pos+=1
    if pos < cfg.edit_cap:
        ec[pos]=(3,0,0,0,0); mask[pos]=True
        en[pos]=[1,decoded.prefix/719,len(decoded.rules)/8,0,0,0,0,0,0,0,0,heuristic,hrank,subtree/40]
    return ec,en,mask,heuristic


def build_case_features(row: Row, decoded: DecodedGenome, cfg: EditFormerConfig, cache, cache_arr, case_meta, lookup):
    nc = row.nc
    probes = np.asarray(cache.probes, np.int32)
    registry = cache_arr[:nc, :, :cache.registry_width]
    match_any = np.zeros((nc, len(probes)), bool)
    best_slack = np.full((nc, len(probes)), -8.0, np.float32)
    first_sem_vals = np.zeros((nc, len(probes)), np.float32)
    chosen_route = np.zeros((nc, len(probes)), np.int16)
    available = np.broadcast_to(probes[None, :] >= decoded.prefix, (nc, len(probes))).copy()
    unmatched = available.copy()
    for rr in decoded.rules:
        rm = available.copy()
        slack = np.full((nc,len(probes)), 8.0, np.float32)
        sem_vals = np.zeros((nc,len(probes)), np.float32)
        sem_seen = False
        for fid, cmp_, idx, val in rr["conds"]:
            off = lookup.get((fid, idx))
            if off is None:
                off = lookup.get((fid, 0))
            if off is None:
                rm &= False; slack = np.minimum(slack, -8.0); continue
            vals = registry[:,:,off]
            rm &= _cmp(vals, cmp_, val)
            slack = np.minimum(slack, _slack(vals.astype(np.float32), cmp_, val))
            if FEATURE_NAME.get(fid) != "STEP" and not sem_seen:
                sem_vals = np.vectorize(signed_log, otypes=[np.float32])(vals)
                sem_seen = True
        fires = rm & unmatched
        chosen_route = np.where(fires, rr["route"], chosen_route)
        first_sem_vals = np.where(fires, sem_vals, first_sem_vals)
        match_any |= fires
        best_slack = np.maximum(best_slack, slack)
        unmatched &= ~fires
    activation_counts = match_any.sum(1)
    # Current140 has one exact target probe; select activated cases first then hardest parent cases.
    if nc > cfg.case_cap:
        order = sorted(range(nc), key=lambda i: (activation_counts[i] == 0, row.baseline[i], i))[:cfg.case_cap]
    else:
        order = list(range(nc))
    cf = np.zeros((cfg.case_cap, CASE_DIM), np.float32)
    cm = np.zeros(cfg.case_cap, bool)
    case_ids = np.full(cfg.case_cap, -1, np.int16)
    target_delta = np.zeros(cfg.case_cap, np.float32)
    target_transition = np.zeros(cfg.case_cap, np.int8)
    man = registry_manifest(); name_off = {(name,args[0] if args else 0):i for i,(name,args) in enumerate(man) if len(args)<=1}
    def val(name, idx, pi):
        off=name_off.get((name,idx),name_off.get((name,0)))
        return float(registry[ci,pi,off]) if off is not None else 0.0
    for oi, ci in enumerate(order):
        m = match_any[ci]
        act = np.flatnonzero(m)
        if len(act): pi=int(act[0]); first=int(probes[pi]); last=int(probes[int(act[-1])])
        else:
            pi=int(np.argmax(best_slack[ci])); first=last=-1
        base=float(row.baseline[ci]); delta=float(row.margins[ci]-row.baseline[ci])
        cf[oi] = [
            math.tanh(base/20000), float(base>0), float(base<0), float(case_meta[ci]["seat"]),
            float(bool(len(act))), float(len(act)/max(1,len(probes))), first/719 if first>=0 else -1, last/719 if last>=0 else -1,
            float(np.max(best_slack[ci])), float(np.mean(best_slack[ci])), float(first_sem_vals[ci,pi]),
            math.tanh(val("OWN_MONEY",0,pi)/20000), math.tanh(val("OPP_MONEY",0,pi)/20000),
            val("OWN_UNITS",0,pi)/40, val("OPP_UNITS",0,pi)/40, val("OWN_LAND",0,pi)/100, val("OPP_LAND",0,pi)/100,
            val("OWN_HIRES_TODAY",0,pi)/40, val("OPP_HIRES_TODAY",0,pi)/40,
            chosen_route[ci,pi]/214 if chosen_route[ci,pi] else 0,
            float(cache.bb_width>19 and cache_arr[ci,pi,cache.registry_width+19] != 0), probes[pi]/719,
        ]
        cm[oi]=True; case_ids[oi]=ci; target_delta[oi]=math.tanh(delta/5000)
        target_transition[oi]=1 if base<0 and row.margins[ci]>0 else (-1 if base>0 and row.margins[ci]<=0 else 0)
    return cf,cm,case_ids,target_delta,target_transition


def dataset_hash(rows: list[Row], ids: list[int]) -> str:
    h=hashlib.sha256()
    for i in sorted(ids, key=lambda j: rows[j].blob_sha):
        r=rows[i]; h.update(bytes.fromhex(r.blob_sha)); h.update(r.campaign.encode()); h.update(r.baseline.tobytes()); h.update(r.margins.tobytes())
    return h.hexdigest()


def _feature_arrays(rows: list[Row], cfg: EditFormerConfig, limit: int | None = None):
    if limit is not None: rows = rows[:limit]
    hist_cache,harr=capture_matrix(HIST_R0); cur_cache,carr=capture_matrix(CURRENT_R0)
    if hist_cache.cases != 40 or hist_cache.probes != tuple(range(0,720,8)) or hist_cache.bb_width != 4: raise ValueError("historical activation cache parent/probe mismatch")
    if cur_cache.cases != 140 or cur_cache.probes != tuple(range(264,288)) or cur_cache.bb_width != 23: raise ValueError("current activation cache parent/probe mismatch")
    lookup=_registry_lookup(); hmeta=_case_meta_historical(40); cmeta=_case_meta_current()
    n=len(rows)
    arrays={
      "tree_cat":np.zeros((n,cfg.tree_cap,TREE_CAT_COLS),np.int16),"tree_num":np.zeros((n,cfg.tree_cap,TREE_NUM_COLS),np.float16),
      "tree_mask":np.zeros((n,cfg.tree_cap),bool),"tree_parent":np.full((n,cfg.tree_cap),-1,np.int16),"tree_depth":np.zeros((n,cfg.tree_cap),np.int8),"tree_rule":np.zeros((n,cfg.tree_cap),np.int8),
      "edit_cat":np.zeros((n,cfg.edit_cap,EDIT_CAT_COLS),np.int16),"edit_num":np.zeros((n,cfg.edit_cap,EDIT_NUM_COLS),np.float16),"edit_mask":np.zeros((n,cfg.edit_cap),bool),
      "case_features":np.zeros((n,cfg.case_cap,CASE_DIM),np.float16),"case_mask":np.zeros((n,cfg.case_cap),bool),"case_ids":np.full((n,cfg.case_cap),-1,np.int16),
      "case_delta":np.zeros((n,cfg.case_cap),np.float16),"case_transition":np.zeros((n,cfg.case_cap),np.int8),
      "heuristic":np.zeros(n,np.float32),"labels":np.zeros((n,9),np.float32),"group":np.zeros(n,np.int16),
      "blob_sha":np.zeros(n,dtype="S64"),"semantic_id":np.zeros(n,dtype="S64"),"parent_kind":np.zeros(n,np.int8),
    }
    groups={name:i for i,name in enumerate(sorted({r.campaign for r in rows}))}
    for i,r in enumerate(rows):
        dec=decode_genome(r.blob)
        tc,tn,tm,tp,td,tr=build_tree(dec,r.parent_kind,cfg); ec,en,em,heur=build_edit(dec,r,cfg)
        cache,arr,meta=(hist_cache,harr,hmeta) if r.parent_kind==1 else (cur_cache,carr,cmeta)
        cf,cm,cids,cd,ct=build_case_features(r,dec,cfg,cache,arr,meta,lookup)
        arrays["tree_cat"][i]=tc;arrays["tree_num"][i]=tn;arrays["tree_mask"][i]=tm;arrays["tree_parent"][i]=tp;arrays["tree_depth"][i]=td;arrays["tree_rule"][i]=tr
        arrays["edit_cat"][i]=ec;arrays["edit_num"][i]=en;arrays["edit_mask"][i]=em;arrays["case_features"][i]=cf;arrays["case_mask"][i]=cm;arrays["case_ids"][i]=cids;arrays["case_delta"][i]=cd;arrays["case_transition"][i]=ct
        arrays["heuristic"][i]=heur; l=r.labels; arrays["labels"][i]=[l["regression"],l["flip"],l["margin_target"],l["worst_target"],l["rank_target"],l["safe"],l["l2w"],l["w2l"],l["mean_delta"]]
        arrays["group"][i]=groups[r.campaign]; arrays["blob_sha"][i]=r.blob_sha.encode(); arrays["semantic_id"][i]=r.semantic_id.encode(); arrays["parent_kind"][i]=r.parent_kind
    return arrays


def feature_arrays_from_capture(rows: list[Row], cfg: EditFormerConfig, capture_path: str | Path,
                                case_meta: list[dict]) -> dict[str, np.ndarray]:
    """Build normal EditFormer features against an explicit frozen public-state capture.

    This is used for fresh failure goals. `case_meta` contains bookkeeping/seat flags only;
    opponent/team/seed identity is never embedded by `build_case_features`.
    """
    cache, arr = capture_matrix(Path(capture_path))
    if cache.cases != len(case_meta):
        raise ValueError(f"capture/case-meta mismatch: {cache.cases} != {len(case_meta)}")
    if any(int(r.nc) != cache.cases for r in rows):
        raise ValueError("every external-context row must span the full capture case panel")
    lookup = _registry_lookup()
    n = len(rows)
    arrays = {
      "tree_cat":np.zeros((n,cfg.tree_cap,TREE_CAT_COLS),np.int16),"tree_num":np.zeros((n,cfg.tree_cap,TREE_NUM_COLS),np.float16),
      "tree_mask":np.zeros((n,cfg.tree_cap),bool),"tree_parent":np.full((n,cfg.tree_cap),-1,np.int16),"tree_depth":np.zeros((n,cfg.tree_cap),np.int8),"tree_rule":np.zeros((n,cfg.tree_cap),np.int8),
      "edit_cat":np.zeros((n,cfg.edit_cap,EDIT_CAT_COLS),np.int16),"edit_num":np.zeros((n,cfg.edit_cap,EDIT_NUM_COLS),np.float16),"edit_mask":np.zeros((n,cfg.edit_cap),bool),
      "case_features":np.zeros((n,cfg.case_cap,CASE_DIM),np.float16),"case_mask":np.zeros((n,cfg.case_cap),bool),"case_ids":np.full((n,cfg.case_cap),-1,np.int16),
      "case_delta":np.zeros((n,cfg.case_cap),np.float16),"case_transition":np.zeros((n,cfg.case_cap),np.int8),
      "heuristic":np.zeros(n,np.float32),"labels":np.zeros((n,9),np.float32),"group":np.zeros(n,np.int16),
      "blob_sha":np.zeros(n,dtype="S64"),"semantic_id":np.zeros(n,dtype="S64"),"parent_kind":np.zeros(n,np.int8),
    }
    groups={name:i for i,name in enumerate(sorted({r.campaign for r in rows}))}
    for i,r in enumerate(rows):
        dec=decode_genome(r.blob)
        tc,tn,tm,tp,td,tr=build_tree(dec,r.parent_kind,cfg);ec,en,em,heur=build_edit(dec,r,cfg)
        cf,cm,cids,cd,ct=build_case_features(r,dec,cfg,cache,arr,case_meta,lookup)
        arrays["tree_cat"][i]=tc;arrays["tree_num"][i]=tn;arrays["tree_mask"][i]=tm;arrays["tree_parent"][i]=tp;arrays["tree_depth"][i]=td;arrays["tree_rule"][i]=tr
        arrays["edit_cat"][i]=ec;arrays["edit_num"][i]=en;arrays["edit_mask"][i]=em;arrays["case_features"][i]=cf;arrays["case_mask"][i]=cm;arrays["case_ids"][i]=cids;arrays["case_delta"][i]=cd;arrays["case_transition"][i]=ct
        arrays["heuristic"][i]=heur;l=r.labels;arrays["labels"][i]=[l["regression"],l["flip"],l["margin_target"],l["worst_target"],l["rank_target"],l["safe"],l["l2w"],l["w2l"],l["mean_delta"]]
        arrays["group"][i]=groups[r.campaign];arrays["blob_sha"][i]=r.blob_sha.encode();arrays["semantic_id"][i]=r.semantic_id.encode();arrays["parent_kind"][i]=r.parent_kind
    return arrays


def prepare_feature_caches(cfg: EditFormerConfig, *, limit: int | None = None, force: bool = False) -> dict:
    OUT.mkdir(parents=True,exist_ok=True); fdir=OUT/"features";fdir.mkdir(exist_ok=True)
    hist,raw=history_rows(); cur=current_rows(); ht,hv=stratified_split(hist);ct,cv=stratified_split(cur)
    suffix=f"smoke{limit}" if limit else "full"
    hp=fdir/f"historical_{suffix}.npz"; cp=fdir/f"current_{suffix}.npz"
    if force or not hp.exists(): np.savez(hp,**_feature_arrays(hist,cfg,limit))
    if force or not cp.exists(): np.savez(cp,**_feature_arrays(cur,cfg,limit))
    source_hashes={
      "historical_train":dataset_hash(hist,ht),"historical_val":dataset_hash(hist,hv),"current_train":dataset_hash(cur,ct),"current_holdout":dataset_hash(cur,cv),
      "historical_activation":sha256_file(HIST_R0),"current_activation":sha256_file(CURRENT_R0),"historical_case_manifest":sha256_file(HIST40_CASES),"current_case_manifest":sha256_file(CURRENT_CASES),
      "current_genomes":sha256_file(MIN_GRU_OUT/"single_predicates.b4gm"),"current_results":sha256_file(MIN_GRU_OUT/"single_predicates_current140.b4rs"),
      "historical_feature_cache":sha256_file(hp),"current_feature_cache":sha256_file(cp),
    }
    manifest={"schema":1,"raw_historical":raw,"historical_unique":len(hist),"historical_train":len(ht),"historical_val":len(hv),"current":len(cur),"current_train":len(ct),"current_holdout":len(cv),
              "historical_safe":sum(r.labels["safe"] for r in hist),"current_safe":sum(r.labels["safe"] for r in cur),"current_holdout_safe":sum(cur[i].labels["safe"] for i in cv),
              "hashes":source_hashes,"leakage":leakage_audit(hist,ht,hv),"historical_cache":str(hp),"current_cache":str(cp),"limit":limit}
    (fdir/f"manifest_{suffix}.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return manifest


def load_cache(kind: str, full: bool = True, limit: int | None = None):
    suffix="full" if full else f"smoke{limit}"
    p=OUT/"features"/f"{kind}_{suffix}.npz"
    if not p.exists(): raise FileNotFoundError(f"missing feature cache {p}; run prepare-data first")
    return dict(np.load(p,allow_pickle=False))


def split_indices(kind: str, rows: list[Row], split: str="primary"):
    return stratified_split(rows) if split=="primary" else grouped_secondary_split(rows)


def numeric_batch(cache: dict, ids: np.ndarray) -> dict[str,np.ndarray]:
    keys=("tree_cat","tree_num","tree_mask","tree_parent","tree_depth","tree_rule","edit_cat","edit_num","edit_mask","case_features","case_mask","case_delta","heuristic","labels","group")
    return {k:cache[k][ids] for k in keys}
