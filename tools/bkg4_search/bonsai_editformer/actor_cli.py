from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))

from actor_data import ACTOR_DATA_DIR, load_actor_dataset, prepare_actor_dataset, row_genome
from actor_generate import (
    actor_generate,
    critic_score_candidates,
    empirical_generate,
    effect_aware_shortlist,
    heuristic_generate,
    random_typed_generate,
    read_candidates,
    write_candidates,
)
from actor_grammar import EDIT_OPERATIONS, FEATURES, OP_INDEX, apply_edit
from actor_model import (
    BonsaiEditActorCritic,
    actor_config_report,
    actor_param_mask,
    critic_param_digest,
    merge_critic_params,
)
from cli import environment_info, load_checkpoint, params_digest, save_checkpoint, tree_all_finite
from data import OUT, ROUTES, current_rows, history_rows, load_cache
from model import BonsaiEditFormer, config_for_size, count_params
from tools.bkg4_search.cuda_io import load_results, pack_genomes
from tools.bkg4_search.module_codec import ModuleGenome, distill_module, genome_from_json

ARCH = "BonsaiEditFormer-ActorCritic-v1"
EXPECTED_PARAMS_5M = 5_581_831
DEFAULT_BASE_SEMANTIC = "b3714ce680b84a74233b053c5bc32f48a37e3c257b46b4e131a0f784b4534286"
NUMERIC_KEYS = ("tree_cat","tree_num","tree_mask","tree_parent","tree_depth","tree_rule","edit_cat","edit_num","edit_mask","case_features","case_mask","case_delta","heuristic","labels","group")


def _to_jax(batch):
    import jax.numpy as jnp
    ints = {"tree_cat","edit_cat","tree_parent","tree_depth","tree_rule","group","actor_op","actor_target",
            "feature","cmp","route","index","feature2","cmp2","index2","condition_count","priority"}
    bools = {"tree_mask","edit_mask","case_mask","actor_target_mask"}
    return {k:jnp.asarray(v, jnp.int32 if k in ints else bool if k in bools else jnp.float32) for k,v in batch.items()}


def actor_batch(cache, ds, ids):
    ids = np.asarray(ids, np.int32); parent = ds["parent_idx"][ids]
    b = {k:cache[k][parent] for k in NUMERIC_KEYS}
    for k in ("actor_op","actor_target","actor_target_mask","feature","cmp","route","index","feature2","cmp2","index2",
              "condition_count","priority","value_norm","value2_norm","prefix_norm","positive","weight","advantage"):
        b[k] = ds[k][ids]
    return _to_jax(b)


def init_actor(cfg, batch, seed=73):
    import jax
    model = BonsaiEditActorCritic(cfg); key = jax.random.PRNGKey(seed); key, drop = jax.random.split(key)
    params = model.init({"params":key,"dropout":drop}, batch, train=False)["params"]
    return model, params


def import_critic(cfg, batch, critic_checkpoint, seed=73):
    model, actor_params = init_actor(cfg, batch, seed)
    critic = load_checkpoint(critic_checkpoint)["params"]
    merged, report = merge_critic_params(actor_params, critic)
    return model, merged, critic, report


def _ce(logits, target):
    import optax
    return optax.softmax_cross_entropy_with_integer_labels(logits, target)


def actor_loss(outputs, batch):
    import jax.numpy as jnp, optax
    pos = batch["positive"]; weight = batch["weight"]; pw = pos * weight; denom = jnp.maximum(jnp.sum(pw), 1.0)
    op = batch["actor_op"]; target = batch["actor_target"]
    target_logits = outputs["actor_target_logits_all"][jnp.arange(op.shape[0]), op]
    losses = {
        "op": jnp.sum(_ce(outputs["actor_op_logits"], op) * pw) / denom,
        "target": jnp.sum(_ce(target_logits, target) * pw) / denom,
    }
    semantic_ops = jnp.isin(op, jnp.asarray([OP_INDEX[x] for x in ("ADD_SPECIALIST","ADD_CONDITION","INSERT_RECOVERY","SPLIT_LANE","COMPOSE_PREDICATES","REPLACE_SUBTREE")]))
    route_ops = jnp.isin(op, jnp.asarray([OP_INDEX[x] for x in ("ADD_SPECIALIST","INSERT_RECOVERY","SPLIT_LANE","CHANGE_ROUTE","REPLACE_MACRO","REPLACE_SUBTREE")]))
    value_ops = semantic_ops | (op == OP_INDEX["CHANGE_THRESHOLD"])
    priority_ops = op == OP_INDEX["CHANGE_PRIORITY"]
    prefix_ops = op == OP_INDEX["CHANGE_WINDOW"]
    def masked(loss_vec, mask):
        w = pw * mask.astype(jnp.float32); return jnp.sum(loss_vec * w) / jnp.maximum(jnp.sum(w), 1.0)
    losses["feature"] = masked(_ce(outputs["actor_feature_logits"], batch["feature"]), semantic_ops)
    losses["cmp"] = masked(_ce(outputs["actor_cmp_logits"], batch["cmp"]), semantic_ops)
    losses["index"] = masked(_ce(outputs["actor_index_logits"], batch["index"]), semantic_ops)
    losses["route"] = masked(_ce(outputs["actor_route_logits"], batch["route"]), route_ops)
    losses["condition_count"] = masked(_ce(outputs["actor_condition_count_logits"], batch["condition_count"]), semantic_ops)
    losses["feature2"] = masked(_ce(outputs["actor_feature2_logits"], batch["feature2"]), semantic_ops & (batch["condition_count"] > 0))
    losses["cmp2"] = masked(_ce(outputs["actor_cmp2_logits"], batch["cmp2"]), semantic_ops & (batch["condition_count"] > 0))
    losses["index2"] = masked(_ce(outputs["actor_index2_logits"], batch["index2"]), semantic_ops & (batch["condition_count"] > 0))
    losses["value"] = masked(optax.huber_loss(outputs["actor_value_norm"], batch["value_norm"], delta=0.15), value_ops)
    losses["value2"] = masked(optax.huber_loss(outputs["actor_value2_norm"], batch["value2_norm"], delta=0.15), semantic_ops & (batch["condition_count"] > 0))
    losses["priority"] = masked(_ce(outputs["actor_priority_logits"], batch["priority"]), priority_ops)
    losses["prefix"] = masked(optax.huber_loss(outputs["actor_prefix_norm"], batch["prefix_norm"], delta=0.15), prefix_ops)
    quality_target = pos.astype(jnp.float32)
    losses["quality"] = jnp.sum(optax.sigmoid_binary_cross_entropy(outputs["actor_proposal_quality_logit"], quality_target) * weight) / jnp.maximum(jnp.sum(weight), 1.0)
    # Known harmful edits reduce probability of the exact operation and target in context.
    neg = (1.0 - pos) * jnp.minimum(weight, 4.0); nden = jnp.maximum(jnp.sum(neg), 1.0)
    op_p = jnp.exp(-_ce(outputs["actor_op_logits"], op)); target_p = jnp.exp(-_ce(target_logits, target))
    losses["negative"] = jnp.sum((-jnp.log(jnp.maximum(1e-5, 1.0 - op_p * target_p))) * neg) / nden
    total = losses["op"] + losses["target"] + 0.8*(losses["feature"]+losses["cmp"]+losses["route"]+losses["value"]) + \
            0.35*(losses["index"]+losses["condition_count"]+losses["feature2"]+losses["cmp2"]+losses["index2"]+losses["value2"]) + \
            0.5*(losses["priority"]+losses["prefix"]) + 0.5*losses["quality"] + 0.15*losses["negative"]
    return total, losses


def actor_metrics(outputs, batch):
    op = np.asarray(batch["actor_op"]); pos = np.asarray(batch["positive"]) > 0
    pred_op = np.argmax(np.asarray(outputs["actor_op_logits"]), axis=-1)
    tl = np.asarray(outputs["actor_target_logits_all"])[np.arange(len(op)), op]; pred_target = np.argmax(tl, axis=-1)
    return {
        "n": int(len(op)), "positive": int(pos.sum()),
        "op_accuracy_positive": float(np.mean(pred_op[pos] == op[pos])) if np.any(pos) else 0.0,
        "target_accuracy_positive": float(np.mean(pred_target[pos] == np.asarray(batch["actor_target"])[pos])) if np.any(pos) else 0.0,
        "quality_positive_mean": float(np.mean(1/(1+np.exp(-np.asarray(outputs["actor_proposal_quality_logit"])[pos])))) if np.any(pos) else 0.0,
        "quality_negative_mean": float(np.mean(1/(1+np.exp(-np.asarray(outputs["actor_proposal_quality_logit"])[~pos])))) if np.any(~pos) else 0.0,
    }


def make_actor_tx(params, lr, wd, clip):
    import jax, optax
    mask = actor_param_mask(params)
    labels = jax.tree.map(lambda m: "actor" if bool(m) else "frozen", mask)
    return optax.multi_transform({"actor":optax.chain(optax.clip_by_global_norm(clip),optax.adamw(lr,weight_decay=wd)),"frozen":optax.set_to_zero()}, labels)


def cmd_prepare(args):
    report = prepare_actor_dataset(Path(args.out), limit_rows=args.limit_rows, max_transitions=args.max_transitions)
    print(json.dumps(report, indent=2, sort_keys=True))


def _load_train_context(args):
    ds = load_actor_dataset(args.dataset); cache = load_cache("historical", True); train = ds["train_idx"]; val = ds["val_idx"]
    sample = actor_batch(cache, ds, train[:min(args.batch_size, len(train))]); cfg = config_for_size("5m", attention_impl=args.attention_impl)
    return ds, cache, train, val, sample, cfg


def cmd_init_critic(args):
    ds,cache,train,val,sample,cfg = _load_train_context(args); model,params,critic,merge = import_critic(cfg,sample,args.critic_checkpoint,args.seed)
    count = count_params(params)
    if count != EXPECTED_PARAMS_5M: raise ValueError(f"ActorCritic param mismatch {count} != {EXPECTED_PARAMS_5M}")
    out = model.apply({"params":params}, sample, train=False); critic_model = BonsaiEditFormer(cfg); co = critic_model.apply({"params":critic}, sample, train=False)
    keys=("regression_logit","flip_logit","margin","worst_win","rank_residual","rank","case_delta")
    exact=all(np.array_equal(np.asarray(out[k]),np.asarray(co[k])) for k in keys)
    if not exact: raise ValueError("critic outputs changed after Actor-Critic import")
    meta={"architecture":ARCH,"parameter_count":count,"critic_checkpoint":str(Path(args.critic_checkpoint).resolve()),"critic_checkpoint_params_digest":params_digest(critic),"merge":merge,"critic_output_exact":exact,"environment":environment_info()}
    save_checkpoint(Path(args.out),params,None,meta)
    print(json.dumps(meta,indent=2,sort_keys=True))


def cmd_smoke(args):
    ds,cache,train,val,sample,cfg = _load_train_context(args); model,params,critic,merge = import_critic(cfg,sample,args.critic_checkpoint,args.seed)
    if count_params(params) != EXPECTED_PARAMS_5M: raise ValueError("parameter count")
    before_critic = critic_param_digest(params)
    import jax, optax
    tx=make_actor_tx(params,args.lr,args.weight_decay,args.clip); state=tx.init(params)
    @jax.jit
    def step(p,s,b,key):
        def obj(q):
            o=model.apply({"params":q},b,train=True,rngs={"dropout":key}); return actor_loss(o,b)
        (loss,parts),grads=jax.value_and_grad(obj,has_aux=True)(p); updates,s=tx.update(grads,s,p); p=optax.apply_updates(p,updates); return p,s,loss,parts,optax.global_norm(grads)
    batch=actor_batch(cache,ds,train[:min(args.batch_size,len(train))]); key=jax.random.PRNGKey(args.seed+99); losses=[]; norms=[]
    for _ in range(args.steps): key,sk=jax.random.split(key);params,state,l,parts,gn=step(params,state,batch,sk);l.block_until_ready();losses.append(float(l));norms.append(float(gn))
    after_critic=critic_param_digest(params); out=model.apply({"params":params},batch,train=False); m=actor_metrics(out,{k:np.asarray(v) for k,v in batch.items()})
    # Grammar/pack/compiler smoke: apply representative generated typed edits and pack them deterministically.
    parent=row_genome(history_rows()[0][int(ds["parent_idx"][train[0]])]); generated=random_typed_generate(parent,n=32,seed=args.seed)
    genomes=[genome_from_json(x.genome) for x in generated]; pack=OUT/"actor/smoke_generated.b4gm";pack_genomes(genomes,pack);pack_sha=hashlib.sha256(pack.read_bytes()).hexdigest()
    ck=OUT/"actor/smoke_actor_checkpoint";save_checkpoint(ck,params,state,{"architecture":ARCH,"smoke":True}); restored=load_checkpoint(ck)["params"]
    report={"architecture":actor_config_report(cfg,params),"merge":merge,"losses":losses,"loss_decreased":losses[-1]<losses[0],"finite_gradients":bool(np.isfinite(norms).all()),"gradient_norms":norms,
            "critic_frozen_exact":before_critic==after_critic,"checkpoint_roundtrip":params_digest(params)==params_digest(restored),"actor_metrics":m,
            "generated_valid":len(generated),"generated_unique":len({x.semantic_id for x in generated}),"packed_genomes_sha256":pack_sha,"environment":environment_info()}
    print(json.dumps(report,indent=2,sort_keys=True))
    if not(report["loss_decreased"] and report["finite_gradients"] and report["critic_frozen_exact"] and report["checkpoint_roundtrip"] and len(generated)>=16): raise SystemExit(2)


def cmd_train(args):
    ds,cache,train,val,sample,cfg=_load_train_context(args);model,params=init_actor(cfg,sample,args.seed); init=load_checkpoint(args.checkpoint)
    # Checkpoint may be critic-initialized Actor-Critic or resumable actor state.
    params=init["params"]; tx=make_actor_tx(params,args.lr,args.weight_decay,args.clip);state=tx.init(params); start_updates=0
    if args.resume and "opt_state" in init: state=init["opt_state"]; start_updates=int(np.asarray(init.get("updates",0)))
    frozen_digest=critic_param_digest(params)
    import jax,optax
    @jax.jit
    def step(p,s,b,key):
        def obj(q):
            o=model.apply({"params":q},b,train=True,rngs={"dropout":key}); return actor_loss(o,b)
        (loss,parts),grads=jax.value_and_grad(obj,has_aux=True)(p);updates,s=tx.update(grads,s,p);p=optax.apply_updates(p,updates);return p,s,loss,parts,optax.global_norm(grads)
    rng=np.random.default_rng(args.seed); key=jax.random.PRNGKey(args.seed+1000); outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True);history=[];updates=start_updates;t0=time.perf_counter()
    def evaluate(epoch,tag):
        ids=val[:min(args.eval_samples,len(val))];b=actor_batch(cache,ds,ids);o=model.apply({"params":params},b,train=False);loss,parts=actor_loss(o,b);rec={"epoch":epoch,"tag":tag,"loss":float(loss),"parts":{k:float(v) for k,v in parts.items()},"metrics":actor_metrics(o,{k:np.asarray(v) for k,v in b.items()})};print("METRIC",json.dumps(rec,sort_keys=True),flush=True);return rec
    history.append(evaluate(0,"init"))
    for ep in range(1,args.epochs+1):
        order=rng.permutation(train);el=[];eg=[]
        for s in range(0,len(order),args.batch_size):
            ids=order[s:s+args.batch_size]
            if len(ids)<args.batch_size: ids=np.resize(ids,args.batch_size)
            b=actor_batch(cache,ds,ids);key,sk=jax.random.split(key);params,state,l,parts,gn=step(params,state,b,sk);l.block_until_ready();el.append(float(l));eg.append(float(gn));updates+=1
        print(json.dumps({"phase":"actor_warmstart","epoch":ep,"loss":float(np.mean(el)),"grad_norm":float(np.mean(eg)),"updates":updates}),flush=True)
        if ep in (args.early_epoch,args.middle_epoch,args.epochs):
            tag="early" if ep==args.early_epoch else "middle" if ep==args.middle_epoch else "final";history.append(evaluate(ep,tag));save_checkpoint(outdir/tag,params,state,{"architecture":ARCH,"epoch":ep,"updates":updates,"critic_frozen_digest":frozen_digest},extra={"updates":jax.numpy.asarray(updates,jax.numpy.int32),"rng":key})
    if critic_param_digest(params)!=frozen_digest: raise ValueError("critic/shared encoder changed during frozen actor warm-start")
    result={"architecture":ARCH,"params":count_params(params),"epochs":args.epochs,"updates":updates,"seconds":time.perf_counter()-t0,"critic_frozen_exact":True,"critic_digest":frozen_digest,"dataset":str(Path(args.dataset).resolve()),"dataset_sha256":hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),"history":history,"environment":environment_info(),"config":vars(args)|{"fn":None}}
    result["config"].pop("fn",None);(outdir/"actor_training_result.json").write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+"\n");print("RESULT",json.dumps(result,sort_keys=True,default=str))


def _current_parent(semantic_id):
    rows=current_rows();cache=load_cache("current",True)
    for i,r in enumerate(rows):
        if r.semantic_id==semantic_id:
            return i,row_genome(r),{k:cache[k][i:i+1] for k in NUMERIC_KEYS}
    raise ValueError(f"current parent semantic id not found: {semantic_id}")


def cmd_generation_eval(args):
    cfg=config_for_size("5m",attention_impl=args.attention_impl);_,parent,parent_batch=_current_parent(args.base_semantic);ck=load_checkpoint(args.checkpoint);model=BonsaiEditActorCritic(cfg)
    t=time.perf_counter();cand=actor_generate(model,ck["params"],cfg,parent,parent_batch,n=args.samples,seed=args.seed,temperature=args.temperature,topk=args.topk,diversity_penalty=args.diversity_penalty);seconds=time.perf_counter()-t
    ops=Counter(x.action.get("operation") for x in cand);targets=len({(x.action.get("operation"),x.action.get("target_node")) for x in cand})
    report={"requested":args.samples,"valid_unique":len(cand),"valid_unique_rate":len(cand)/max(1,args.samples),"semantic_duplicate_rate_after_canonicalization":1-len({x.semantic_id for x in cand})/max(1,len(cand)),"operation_distribution":ops,"target_diversity":targets,"novel_vs_parent":sum(x.semantic_id!=parent.semantic_id() for x in cand),"seconds":seconds,"candidates_s":len(cand)/max(seconds,1e-9)}
    write_candidates(args.out,cand);Path(args.report).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2,sort_keys=True))


def cmd_generator_benchmark(args):
    cfg=config_for_size("5m",attention_impl=args.attention_impl);parent_idx,parent,parent_batch=_current_parent(args.base_semantic);ck=load_checkpoint(args.checkpoint);model=BonsaiEditActorCritic(cfg);ds=load_actor_dataset(args.dataset);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    parent_margins=np.asarray(current_rows()[parent_idx].margins,np.int32)
    proposal_n=max(args.cuda_budget,args.proposal_multiplier*args.cuda_budget)
    sources={
      "actor":actor_generate(model,ck["params"],cfg,parent,parent_batch,n=proposal_n,seed=args.seed,temperature=args.temperature,topk=args.topk,diversity_penalty=args.diversity_penalty),
      "heuristic":heuristic_generate(parent,n=proposal_n,seed=args.seed+11),
      "random":random_typed_generate(parent,n=args.cuda_budget,seed=args.seed+23),
      "empirical":empirical_generate(parent,ds,n=args.cuda_budget,seed=args.seed+37),
    }
    actor_scored=critic_score_candidates(model,ck["params"],cfg,sources["actor"],parent=parent,parent_margins=parent_margins,batch_size=args.critic_batch);heur_scored=critic_score_candidates(model,ck["params"],cfg,sources["heuristic"],parent=parent,parent_margins=parent_margins,batch_size=args.critic_batch)
    actor_filtered,actor_diag=effect_aware_shortlist(actor_scored,args.cuda_budget,max_regression_prob=args.max_regression_prob,exploration_regression_prob=args.exploration_regression_prob,exploration_fraction=args.exploration_fraction,diversity_penalty=args.operation_diversity_penalty)
    heuristic_filtered,heuristic_diag=effect_aware_shortlist(heur_scored,args.cuda_budget,max_regression_prob=args.max_regression_prob,exploration_regression_prob=args.exploration_regression_prob,exploration_fraction=args.exploration_fraction,diversity_penalty=args.operation_diversity_penalty)
    # Persist the full scored pools, not only the CUDA shortlist. This makes false-positive
    # no-op/effect analysis reproducible without rerunning expensive actor generation.
    write_candidates(out/"actor_pool_scored.jsonl",actor_scored);write_candidates(out/"heuristic_pool_scored.jsonl",heur_scored)
    lanes={
      "actor_raw":actor_scored[:args.cuda_budget],"actor_critic":actor_filtered,
      "heuristic_raw":heur_scored[:args.cuda_budget],"heuristic_critic":heuristic_filtered,
      "random_typed":sources["random"][:args.cuda_budget],"empirical_ops":sources["empirical"][:args.cuda_budget],
    }
    parent_path=out/"base_parent.json";parent_path.write_text(json.dumps(parent.canonical(),indent=2,sort_keys=True)+"\n")
    summary={"base_parent":parent.semantic_id(),"proposal_n":proposal_n,"cuda_budget_per_lane":args.cuda_budget,
             "selection":{"actor_critic":actor_diag,"heuristic_critic":heuristic_diag,
                          "max_regression_prob":args.max_regression_prob,"exploration_regression_prob":args.exploration_regression_prob,
                          "exploration_fraction":args.exploration_fraction,"operation_diversity_penalty":args.operation_diversity_penalty},"lanes":{}}
    for name,c in lanes.items():
        write_candidates(out/f"{name}.jsonl",c)
        summary["lanes"][name]={"count":len(c),"unique":len({x.semantic_id for x in c}),"operations":Counter(x.action.get('operation') for x in c),
                                "predicted_strong_effect":sum(float(x.critic_effect_mean_abs or 0)>=.15 for x in c),
                                "predicted_route_divergent":sum(int(x.probe_route_divergence_cases or 0)>0 for x in c),
                                "predicted_loss_targeted":sum(int(x.probe_loss_target_cases or 0)>0 for x in c),
                                "mean_regression_prob":float(np.mean([x.critic_regression_prob for x in c if x.critic_regression_prob is not None])) if any(x.critic_regression_prob is not None for x in c) else None}
    (out/"generator_benchmark.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");print(json.dumps(summary,indent=2,sort_keys=True))


def cmd_prepare_fresh(args):
    rows=[json.loads(x) for x in Path(args.source_jsonl).read_text().splitlines() if x.strip()];seeds=sorted({int(r['seed']) for r in rows});families={name:i for i,name in enumerate(sorted({str(r['team']) for r in rows}))};seat_by_seed={s:set() for s in seeds}
    for r in rows:seat_by_seed[int(r['seed'])].add(int(r['seat']))
    if len(seeds)<args.min_seeds or any(v!={0,1} for v in seat_by_seed.values()):raise ValueError(f"fresh CUDA reward must be both seats across >= {args.min_seeds} seeds: seeds={len(seeds)} seats={seat_by_seed}")
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);lines=[]
    for r in rows:lines.append("\t".join([str(r['tape']),str(int(r['seat'])),str(int(r['seed'])),str(families[str(r['team'])]),str(int(r.get('protected',1))),str(int(r.get('own_history',0))) ]))
    out.write_text("\n".join(lines)+"\n");m={"cases":len(rows),"unique_seeds":len(seeds),"both_seats_per_seed":True,"seeds":seeds,"families":families,"sha256":hashlib.sha256(out.read_bytes()).hexdigest()};Path(args.manifest).write_text(json.dumps(m,indent=2,sort_keys=True)+"\n");print(json.dumps(m,indent=2,sort_keys=True))


def cmd_cuda(args):
    cases_manifest=json.loads(Path(args.cases_manifest).read_text());
    if not cases_manifest.get("both_seats_per_seed") or int(cases_manifest.get("unique_seeds",0))<args.min_seeds:raise ValueError("refusing single-seat/low-seed CUDA reward manifest")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);base=genome_from_json(json.loads(Path(args.base_parent).read_text()));report={"cases":cases_manifest,"lanes":{},"cuda_evaluator":str(Path(args.exe).resolve())}
    for lane_file in sorted(Path(args.lanes).glob("*.jsonl")):
        candidates=read_candidates(lane_file);genomes=[base]+[genome_from_json(c.genome) for c in candidates];gm=out/f"{lane_file.stem}.b4gm";rs=out/f"{lane_file.stem}.b4rs";pack_genomes(genomes,gm)
        env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(args.gpu);cmd=[args.exe,"--controller",args.controller,"--routes",args.routes,"--legacy-routes",args.legacy_routes,"--cases",args.cases,"--genomes",str(gm),"--out",str(rs),"--feed-slot","5","--feed-count","13","--tpb","64"]
        p=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True,check=False);(out/f"{lane_file.stem}.cuda.log").write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError(f"CUDA lane {lane_file.stem} failed: {p.stderr[-3000:]}")
        rr=load_results(rs);baseline=np.asarray([a-b for a,b,_,_ in rr[0]],np.int32);safe=[];recs=[]
        for idx,(cand,row) in enumerate(zip(candidates,rr[1:]),1):
            m=np.asarray([a-b for a,b,_,_ in row],np.int32);l2w=int(np.sum((baseline<0)&(m>0)));w2l=int(np.sum((baseline>0)&(m<=0)));reg3=int(np.sum((baseline>0)&((m-baseline)<-3000)));md=float(np.mean(m-baseline));is_safe=l2w>0 and w2l==0 and reg3==0
            changed_cases=int(np.sum(m!=baseline)); exact_parent_behavior=changed_cases==0
            rec={"candidate_index":idx,"semantic_id":cand.semantic_id,"safe":is_safe,"l2w":l2w,"w2l":w2l,"reg3":reg3,"mean_delta":md,"changed_cases":changed_cases,"exact_parent_behavior":exact_parent_behavior,"source":cand.source,"action":cand.action,
                 "critic_regression_prob":cand.critic_regression_prob,"critic_flip_prob":cand.critic_flip_prob,"critic_margin_pred":cand.critic_margin_pred,"critic_effect_mean_abs":cand.critic_effect_mean_abs,"critic_effect_cases":cand.critic_effect_cases,
                 "probe_activated_cases":cand.probe_activated_cases,"probe_route_divergence_cases":cand.probe_route_divergence_cases,"probe_loss_target_cases":cand.probe_loss_target_cases,"probe_win_exposure_cases":cand.probe_win_exposure_cases,"probe_effect_score":cand.probe_effect_score};recs.append(rec)
            if is_safe:safe.append(rec)
        toks={};
        for tok in p.stdout.split():
            if '=' in tok:
                k,v=tok.split('=',1);toks[k]=v
        lane={"candidates":len(candidates),"games":int(toks.get('games',len(genomes)*len(baseline))),"games_s":float(toks.get('games_s',0)),"safe_count":len(safe),"safe_per_1000_cuda_evaluated":1000*len(safe)/max(1,len(candidates)),"first_safe_candidate":safe[0]['candidate_index'] if safe else None,"loss_to_win_candidates":sum(r['l2w']>0 for r in recs),"win_to_loss_candidates":sum(r['w2l']>0 for r in recs),"positive_margin_candidates":sum(r['mean_delta']>0 for r in recs),"behavior_changing_candidates":sum(not r['exact_parent_behavior'] for r in recs),"exact_parent_behavior_candidates":sum(r['exact_parent_behavior'] for r in recs),"mean_changed_cases":float(np.mean([r['changed_cases'] for r in recs])) if recs else 0.0,"safe_records":safe,"records":recs};report["lanes"][lane_file.stem]=lane
    (out/"cuda_campaign_result.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2,sort_keys=True))


def cmd_package(args):
    run=Path(args.run_dir);required=["actor_training_result.json","generation_evaluation.json","generator_benchmark/generator_benchmark.json","cuda_campaign/cuda_campaign_result.json"];missing=[x for x in required if not(run/x).exists()]
    if missing:raise FileNotFoundError(missing)
    training=json.loads((run/required[0]).read_text());generation=json.loads((run/required[1]).read_text());genbench=json.loads((run/required[2]).read_text());cuda=json.loads((run/required[3]).read_text());manifest={"schema":1,"architecture":ARCH,"params":training['params'],"created_utc":datetime.now(UTC).isoformat(),"critic_frozen_exact":training['critic_frozen_exact'],"actor_training":training,"generation":generation,"generator_benchmark":genbench,"cuda_campaign":cuda,"decision_gate":{"valid_generation":generation['valid_unique_rate']>=0.9,"diverse_operations":len(generation['operation_distribution'])>=3,"new_safe_actor_critic":bool(cuda['lanes'].get('actor_critic',{}).get('safe_count',0))}}
    (run/"result_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,default=str)+"\n");lines=[f"# {ARCH}","",f"- Parameters: **{training['params']:,}**",f"- Critic frozen exact: **{training['critic_frozen_exact']}**",f"- Valid unique actor generation: **{generation['valid_unique_rate']:.3f}**",f"- Operation families generated: **{len(generation['operation_distribution'])}**",""]
    for lane,x in cuda['lanes'].items():lines.append(f"- {lane}: safe={x['safe_count']}, l2w={x['loss_to_win_candidates']}, w2l={x['win_to_loss_candidates']}, games/s={x['games_s']:.1f}")
    (run/"SUMMARY.md").write_text("\n".join(lines)+"\n");package=Path(args.package);package.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(package,"w:gz",compresslevel=6) as tf:
        for p in run.rglob('*'):
            if p.is_file() and ('final/' not in str(p) or p.name.startswith('_CHECKPOINT')):tf.add(p,arcname=f"{run.name}/{p.relative_to(run)}")
    print(json.dumps({"package":str(package),"sha256":hashlib.sha256(package.read_bytes()).hexdigest(),"bytes":package.stat().st_size},indent=2))


def parser():
    ap=argparse.ArgumentParser(description=ARCH);sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('prepare-data');p.add_argument('--out',default=str(ACTOR_DATA_DIR/'transitions_full.npz'));p.add_argument('--limit-rows',type=int);p.add_argument('--max-transitions',type=int,default=60000);p.set_defaults(fn=cmd_prepare)
    def train_common(p):
        p.add_argument('--dataset',required=True);p.add_argument('--attention-impl',choices=('xla','cudnn'),default='cudnn');p.add_argument('--batch-size',type=int,default=512);p.add_argument('--seed',type=int,default=73)
    p=sp.add_parser('init-critic');train_common(p);p.add_argument('--critic-checkpoint',required=True);p.add_argument('--out',required=True);p.set_defaults(fn=cmd_init_critic)
    p=sp.add_parser('smoke');train_common(p);p.add_argument('--critic-checkpoint',required=True);p.add_argument('--steps',type=int,default=4);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--clip',type=float,default=2.0);p.set_defaults(fn=cmd_smoke)
    p=sp.add_parser('train-actor');train_common(p);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=10);p.add_argument('--early-epoch',type=int,default=2);p.add_argument('--middle-epoch',type=int,default=6);p.add_argument('--eval-samples',type=int,default=2048);p.add_argument('--lr',type=float,default=8e-4);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--clip',type=float,default=2.0);p.add_argument('--resume',action='store_true');p.set_defaults(fn=cmd_train)
    def gen_common(p):
        p.add_argument('--checkpoint',required=True);p.add_argument('--attention-impl',choices=('xla','cudnn'),default='cudnn');p.add_argument('--base-semantic',default=DEFAULT_BASE_SEMANTIC);p.add_argument('--seed',type=int,default=73);p.add_argument('--temperature',type=float,default=.9);p.add_argument('--topk',type=int,default=8);p.add_argument('--diversity-penalty',type=float,default=.35)
    p=sp.add_parser('evaluate-generation');gen_common(p);p.add_argument('--samples',type=int,default=512);p.add_argument('--out',required=True);p.add_argument('--report',required=True);p.set_defaults(fn=cmd_generation_eval)
    p=sp.add_parser('generator-benchmark');gen_common(p);p.add_argument('--dataset',required=True);p.add_argument('--out',required=True);p.add_argument('--cuda-budget',type=int,default=128);p.add_argument('--proposal-multiplier',type=int,default=4);p.add_argument('--critic-batch',type=int,default=256);p.add_argument('--max-regression-prob',type=float,default=.35);p.add_argument('--exploration-regression-prob',type=float,default=.999);p.add_argument('--exploration-fraction',type=float,default=.25);p.add_argument('--operation-diversity-penalty',type=float,default=.25);p.set_defaults(fn=cmd_generator_benchmark)
    p=sp.add_parser('prepare-fresh-cases');p.add_argument('--source-jsonl',required=True);p.add_argument('--out',required=True);p.add_argument('--manifest',required=True);p.add_argument('--min-seeds',type=int,default=6);p.set_defaults(fn=cmd_prepare_fresh)
    p=sp.add_parser('cuda-campaign');p.add_argument('--lanes',required=True);p.add_argument('--base-parent',required=True);p.add_argument('--cases',required=True);p.add_argument('--cases-manifest',required=True);p.add_argument('--exe',required=True);p.add_argument('--controller',required=True);p.add_argument('--routes',required=True);p.add_argument('--legacy-routes',required=True);p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--min-seeds',type=int,default=6);p.set_defaults(fn=cmd_cuda)
    p=sp.add_parser('package-results');p.add_argument('--run-dir',required=True);p.add_argument('--package',required=True);p.set_defaults(fn=cmd_package)
    return ap


def main():
    args = parser().parse_args()
    return args.fn(args)

if __name__=='__main__': main()
