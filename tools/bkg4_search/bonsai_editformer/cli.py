from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))

from data import (BASE0_IDENTITY,CURRENT_CASES,CURRENT_IDENTITY,CURRENT_R0,HIST40_CASES,HIST_R0,MIN_GRU_OUT,OUT,
                  current_rows,dataset_hash,history_rows,load_cache,load_r0,numeric_batch,prepare_feature_caches,sha256_file,split_indices,stratified_split)
from model import BonsaiEditFormer,EditFormerConfig,count_params


def stack_imports():
    import jax,jaxlib,flax,optax,orbax.checkpoint as ocp
    return jax,jaxlib,flax,optax,ocp


def repo_meta():
    def cmd(*a): return subprocess.check_output(a,cwd=ROOT,text=True).strip()
    tree=cmd("git","rev-parse","HEAD:purejaxrl")
    return {"repo_head":cmd("git","rev-parse","HEAD"),"branch":cmd("git","branch","--show-current"),"purejaxrl_path":str(ROOT/"purejaxrl"),"purejaxrl_tree_sha":tree,"purejaxrl_nested_git":(ROOT/"purejaxrl/.git").exists()}


def cfg_for(args):
    cfg=EditFormerConfig(attention_impl=getattr(args,"attention_impl","xla"))
    ab=getattr(args,"ablation","complete")
    levels={"t0":0,"t1":1,"t2":2,"t3":3,"t4":4,"t5":5,"complete":5}; lv=levels[ab]
    cfg=replace(cfg,use_structural_bias=lv>=1,use_edit_delta=lv>=2,use_activation_context=lv>=3,use_rank_loss=lv>=4,use_heuristic_residual=lv>=5)
    return cfg


def environment_info():
    jax,jaxlib,flax,optax,ocp=stack_imports(); dev=jax.devices()[0]
    return {**repo_meta(),"jax":jax.__version__,"jaxlib":jaxlib.__version__,"flax":flax.__version__,"optax":optax.__version__,"orbax":getattr(ocp,"__version__","unknown"),
            "backend":jax.default_backend(),"device":str(dev),"platform":dev.platform,"bf16_device":str(jax.numpy.ones((1,),dtype=jax.numpy.bfloat16).device),
            "dpa_signature":str(inspect.signature(jax.nn.dot_product_attention))}


def ensure_activation_cache():
    OUT.mkdir(parents=True,exist_ok=True); exe=ROOT/"scratch/bkg4_search/bkg4_information_state_capture"
    hist_probes=tuple(range(0,720,8)); hist_ok=HIST_R0.exists()
    if hist_ok:
        try: hist_ok=load_r0(HIST_R0).probes==hist_probes
        except Exception: hist_ok=False
    if not hist_ok:
        rows=[]
        for line in HIST40_CASES.read_text().splitlines():
            z=line.split("\t"); rows.append("\t".join([z[1],z[2],z[3],z[5],z[6],z[7],z[10]]))
        cases=OUT/"historical40_cases7.tsv"; probes=OUT/"historical_activation_probes.txt"; truth=OUT/"historical40_truth_unused.b4is"
        cases.write_text("\n".join(rows)+"\n"); probes.write_text("\n".join(str(x) for x in hist_probes)+"\n")
        cmd=[str(exe),"--controller",str(ROOT/"bonsai-kg/integrations/bkg4_champion/assets/controller.bkgc"),"--routes",str(ROOT/"bonsai-kg/integrations/stateful_v6/assets/routes.bkv6"),"--legacy-routes",str(ROOT/"bonsai-kg/integrations/bkg4_champion/assets/legacy_routes.bkv6"),"--cases",str(cases),"--probes",str(probes),"--out",str(HIST_R0),"--truth-out",str(truth),"--feed-slot","0","--feed-count","0","--strategic-route-slot","-1","--tpb","64"]
        subprocess.run(cmd,cwd=ROOT,check=True); truth.unlink(missing_ok=True)
    current_probes=tuple(range(264,288)); current_ok=CURRENT_R0.exists()
    if current_ok:
        try: current_ok=load_r0(CURRENT_R0).probes==current_probes
        except Exception: current_ok=False
    if not current_ok:
        current7=OUT/"current140_cases7.tsv"; probe=OUT/"current_activation_probes.txt"; truth=OUT/"current140_truth_unused.b4is"
        rows=[]
        for line in CURRENT_CASES.read_text().splitlines():
            if not line.strip() or line.startswith("#"): continue
            z=line.split("\t"); rows.append("\t".join([z[0],z[1],z[2],z[3],z[4],z[5],"0"]))
        current7.write_text("\n".join(rows)+"\n"); probe.write_text("\n".join(str(x) for x in current_probes)+"\n")
        asset=ROOT/"bonsai-kg/integrations/bkg4_champion/champions/w33_s172_expansion_v1"
        cmd=[str(exe),"--controller",str(asset/"controller.bkgc"),"--routes",str(asset/"routes.bkv6"),"--legacy-routes",str(asset/"legacy_routes.bkv6"),"--cases",str(current7),"--probes",str(probe),"--out",str(CURRENT_R0),"--truth-out",str(truth),"--feed-slot","5","--feed-count","13","--strategic-route-slot","19","--tpb","64"]
        subprocess.run(cmd,cwd=ROOT,check=True); truth.unlink(missing_ok=True)


def init_model(cfg:EditFormerConfig,batch,seed=73):
    jax,*_=stack_imports(); model=BonsaiEditFormer(cfg); key=jax.random.PRNGKey(seed); key,drop=jax.random.split(key)
    params=model.init({"params":key,"dropout":drop},batch,train=False)["params"]
    return model,params


def to_jax(batch):
    import jax.numpy as jnp
    out={}
    for k,v in batch.items():
        if k in ("tree_cat","edit_cat","tree_parent","tree_depth","tree_rule","group"): out[k]=jnp.asarray(v,dtype=jnp.int32)
        elif k in ("tree_mask","edit_mask","case_mask"): out[k]=jnp.asarray(v,dtype=bool)
        else: out[k]=jnp.asarray(v,dtype=jnp.float32)
    return out


def loss_fn(outputs,batch,cfg,weights):
    import jax.numpy as jnp, jax.nn as jnn, optax
    y=batch["labels"].astype(jnp.float32); sw=batch.get("sample_weight",jnp.ones(y.shape[0],jnp.float32)).astype(jnp.float32); swsum=jnp.maximum(jnp.sum(sw),1.0)
    reg,flip,margin,worst,rank_t,safe=y[:,0],y[:,1],y[:,2],y[:,3],y[:,4],y[:,5]
    def wbce(logit,target,posw):
        base=optax.sigmoid_binary_cross_entropy(logit,target)*(1+(posw-1)*target); return jnp.sum(base*sw)/swsum
    lreg=wbce(outputs["regression_logit"],reg,weights["reg_pos"])
    lflip=wbce(outputs["flip_logit"],flip,weights["flip_pos"])
    lmargin=jnp.sum(optax.huber_loss(outputs["margin"],margin,delta=0.25)*sw)/swsum
    lworst=jnp.sum(optax.huber_loss(outputs["worst_win"],worst,delta=0.25)*sw)/swsum
    pred=outputs["rank"]
    di=rank_t[:,None]-rank_t[None,:]; pd=pred[:,None]-pred[None,:]
    same=batch["group"][:,None]==batch["group"][None,:]; nonzero=jnp.abs(di)>1e-6
    sign=jnp.sign(di); top=1+4*jnp.maximum(safe[:,None],safe[None,:])+2*jnp.maximum(flip[:,None],flip[None,:])
    heur_pair=jnp.maximum(jnp.maximum(batch["heuristic"][:,None],batch["heuristic"][None,:]),0.0)
    hard=1+2*heur_pair*jnp.maximum(reg[:,None],reg[None,:])
    pair_weight=sw[:,None]*sw[None,:]; mask=same&nonzero&(pair_weight>0); pair=jnn.softplus(-sign*pd)*top*hard*pair_weight
    lrank=jnp.sum(jnp.where(mask,pair,0.0))/jnp.maximum(jnp.sum(jnp.where(mask,pair_weight,0.0)),1)
    ctarget=batch["case_delta"].astype(jnp.float32); cm=batch["case_mask"].astype(jnp.float32)*sw[:,None]
    lcase=jnp.sum(optax.huber_loss(outputs["case_delta"],ctarget,delta=0.25)*cm)/jnp.maximum(jnp.sum(cm),1)
    total=weights["reg"]*lreg+weights["flip"]*lflip+weights["margin"]*lmargin+weights["worst"]*lworst+weights["case"]*lcase+(weights["rank"]*lrank if cfg.use_rank_loss else 0)
    return total,{"reg":lreg,"flip":lflip,"margin":lmargin,"worst":lworst,"rank":lrank,"case":lcase}


def default_weights(labels,args=None):
    reg=float(np.mean(labels[:,0])); flip=float(np.mean(labels[:,1]))
    get=lambda name,default:float(getattr(args,name,default)) if args is not None else default
    return {"reg":get("w_reg",6.0),"flip":get("w_flip",4.0),"margin":get("w_margin",1.0),"worst":get("w_worst",3.0),"rank":get("w_rank",6.0),"case":get("w_case",1.5),
            "reg_pos":max(1.0,(1-reg)/max(reg,1e-4)),"flip_pos":max(1.0,(1-flip)/max(flip,1e-4))}


def make_tx(lr,wd,clip):
    import optax
    return optax.chain(optax.clip_by_global_norm(clip),optax.adamw(lr,weight_decay=wd))


def tree_all_finite(tree):
    import jax,jax.numpy as jnp
    leaves=jax.tree.leaves(tree); return bool(np.all([np.asarray(jnp.all(jnp.isfinite(x))) for x in leaves]))


def params_digest(params):
    import jax
    h=hashlib.sha256()
    for x in jax.tree.leaves(params): h.update(np.asarray(x).tobytes())
    return h.hexdigest()


def make_train_step(model,cfg,tx,weights):
    import jax,optax
    @jax.jit
    def step(params,opt_state,batch,key):
        def obj(p):
            out=model.apply({"params":p},batch,train=True,rngs={"dropout":key}); return loss_fn(out,batch,cfg,weights)
        (loss,aux),grads=jax.value_and_grad(obj,has_aux=True)(params); updates,opt_state=tx.update(grads,opt_state,params); params=optax.apply_updates(params,updates)
        gnorm=optax.global_norm(grads); return params,opt_state,loss,aux,gnorm
    return step


def make_epoch_step(model,cfg,tx,weights,batch_size,effective_batch_size,n_ids):
    import jax,jax.numpy as jnp,optax
    if effective_batch_size<batch_size or effective_batch_size%batch_size: raise ValueError("effective batch must be a multiple of microbatch")
    accum_steps=effective_batch_size//batch_size; pad=(-n_ids)%effective_batch_size
    @jax.jit
    def epoch(params,opt_state,data,train_ids,key):
        key,pk=jax.random.split(key); perm=jax.random.permutation(pk,train_ids); valid=jnp.ones((n_ids,),jnp.float32)
        if pad:
            perm=jnp.concatenate([perm,jnp.resize(perm,(pad,))],axis=0); valid=jnp.concatenate([valid,jnp.zeros((pad,),jnp.float32)],axis=0)
        batches=perm.reshape((-1,accum_steps,batch_size)); weights_b=valid.reshape((-1,accum_steps,batch_size))
        def update_body(carry,x):
            params,opt_state,key=carry; micro_ids,micro_weights=x; zero=jax.tree.map(jnp.zeros_like,params); total_valid=jnp.maximum(jnp.sum(micro_weights),1.0)
            def micro_body(mc,xm):
                grad_acc,loss_acc,key=mc; ids,sample_weight=xm; key,sk=jax.random.split(key); batch={k:v[ids] for k,v in data.items()}; batch["sample_weight"]=sample_weight
                def obj(p):
                    out=model.apply({"params":p},batch,train=True,rngs={"dropout":sk}); return loss_fn(out,batch,cfg,weights)
                (loss,_),grads=jax.value_and_grad(obj,has_aux=True)(params); frac=jnp.sum(sample_weight)/total_valid; grad_acc=jax.tree.map(lambda a,g:a+g*frac,grad_acc,grads)
                return (grad_acc,loss_acc+loss*frac,key),None
            (grads,loss,key),_=jax.lax.scan(micro_body,(zero,jnp.array(0.0,jnp.float32),key),(micro_ids,micro_weights))
            updates,opt_state=tx.update(grads,opt_state,params); params=optax.apply_updates(params,updates)
            return (params,opt_state,key),(loss,optax.global_norm(grads))
        (params,opt_state,key),(losses,gnorms)=jax.lax.scan(update_body,(params,opt_state,key),(batches,weights_b))
        return params,opt_state,key,jnp.mean(losses),jnp.mean(gnorms)
    return epoch


def batch_for(cache,ids): return to_jax(numeric_batch(cache,np.asarray(ids,np.int32)))


def score(model,params,cache,ids,batch_size=128):
    import jax,jax.numpy as jnp
    @jax.jit
    def f(b): return model.apply({"params":params},b,train=False)
    outs={}
    for s in range(0,len(ids),batch_size):
        b=batch_for(cache,ids[s:s+batch_size]); o=f(b)
        for k,v in o.items(): outs.setdefault(k,[]).append(np.asarray(v))
    return {k:np.concatenate(v) for k,v in outs.items()}


def rankdata(x):
    order=np.argsort(x,kind="mergesort"); r=np.empty(len(x),float); r[order]=np.arange(len(x)); return r


def metrics(pred,labels,name):
    true=labels[:,4]; safe=labels[:,5]>0; reg=labels[:,0]>0; n=len(true)
    sp=float(np.corrcoef(rankdata(pred),rankdata(true))[0,1]) if n>2 else 0
    rng=np.random.default_rng(123); m=min(20000,max(100,n*4)); a=rng.integers(0,n,m);b=rng.integers(0,n,m);mk=true[a]!=true[b]
    pair=float(np.mean((pred[a[mk]]-pred[b[mk]])*(true[a[mk]]-true[b[mk]])>0)) if np.any(mk) else 0
    order=np.argsort(-pred,kind="mergesort"); out={"name":name,"n":n,"pairwise":pair,"spearman":sp,"safe_prevalence":float(np.mean(safe)),"regression_prevalence":float(np.mean(reg))}
    k10=max(1,n//10); ix=order[:k10]; out["top10_safe_rate"]=float(np.mean(safe[ix])); out["top10_regression_rate"]=float(np.mean(reg[ix]))
    for k in (16,32,64):
        kk=min(k,n); ix=order[:kk]; out[f"precision_safe@{k}"]=float(np.mean(safe[ix])); out[f"recall_safe@{k}"]=float(np.sum(safe[ix])/max(1,np.sum(safe))); out[f"regression@{k}"]=float(np.mean(reg[ix]))
    return out


def save_checkpoint(path,params,opt_state=None,meta=None,extra=None):
    _,_,_,_,ocp=stack_imports(); path=Path(path).resolve(); path.parent.mkdir(parents=True,exist_ok=True)
    ck=ocp.StandardCheckpointer(); state={"params":params}
    if opt_state is not None: state["opt_state"]=opt_state
    if extra: state.update(extra)
    ck.save(path,state,force=True,custom_metadata=meta or {})
    ck.wait_until_finished()
    ck.close()


def load_checkpoint(path,target=None,strict=True):
    _,_,_,_,ocp=stack_imports(); ck=ocp.StandardCheckpointer(); state=ck.restore(Path(path).resolve(),target=target,strict=strict); ck.close(); return state


def cli_config(args):
    return {k:v for k,v in vars(args).items() if k!="fn" and isinstance(v,(str,int,float,bool,type(None)))}


def checkpoint_meta(cfg,manifest,seed,phase,epoch,params,run_config=None,updates=0):
    return {**environment_info(),"model":asdict(cfg),"parameter_count":count_params(params),"seed":seed,"phase":phase,"epoch":epoch,"updates":int(updates),"dataset_hashes":manifest.get("hashes",{}),"run_config":run_config or {}}


def cmd_stack(args):
    jax,*_=stack_imports(); info=environment_info(); cfg=cfg_for(args); info["model_config"]=asdict(cfg)
    # Cheap backend micro-check on real head shape with explicit bias.
    q=jax.numpy.ones((2,cfg.tree_cap,cfg.heads,cfg.head_dim),jax.numpy.bfloat16); bias=jax.numpy.zeros((2,cfg.heads,cfg.tree_cap,cfg.tree_cap),jax.numpy.bfloat16)
    for impl in ("xla","cudnn"):
        t=time.perf_counter()
        try:
            fn=jax.jit(lambda x,b:jax.nn.dot_product_attention(x,x,x,bias=b,implementation=impl)); y=fn(q,bias); y.block_until_ready(); info[f"attention_{impl}"]={"ok":True,"first_seconds":time.perf_counter()-t}
        except Exception as e: info[f"attention_{impl}"]={"ok":False,"error":str(e)[:500]}
    print(json.dumps(info,indent=2,sort_keys=True))


def cmd_prepare(args):
    ensure_activation_cache(); cfg=cfg_for(args); t=time.perf_counter(); m=prepare_feature_caches(cfg,limit=args.limit,force=args.force); m["prepare_seconds"]=time.perf_counter()-t; print(json.dumps(m,indent=2,sort_keys=True))


def smoke_cache(cfg,limit=128):
    ensure_activation_cache(); prepare_feature_caches(cfg,limit=limit,force=False); return load_cache("historical",False,limit),load_cache("current",False,limit)


def cmd_smoke(args):
    import jax,jax.numpy as jnp,optax
    cfg=cfg_for(args); hc,cc=smoke_cache(cfg,args.limit); n=min(args.batch_size,len(hc["labels"])); ids=np.arange(n); batch=batch_for(hc,ids)
    model,params=init_model(cfg,batch,args.seed); params2=init_model(cfg,batch,args.seed)[1]
    count=count_params(params); same=params_digest(params)==params_digest(params2)
    out=model.apply({"params":params},batch,train=False); shapes={k:list(v.shape) for k,v in out.items()}; weights=default_weights(hc["labels"][:n]); loss,parts=loss_fn(out,batch,cfg,weights)
    tx=make_tx(args.lr,args.weight_decay,args.clip); opt_state=tx.init(params); step=make_train_step(model,cfg,tx,weights); before=params_digest(params)
    key=jax.random.PRNGKey(args.seed+1); losses=[]; gnorms=[]
    for i in range(args.steps):
        key,sk=jax.random.split(key); params,opt_state,l,aux,gn=step(params,opt_state,batch,sk); l.block_until_ready(); losses.append(float(l));gnorms.append(float(gn))
    after=params_digest(params)
    ck=OUT/"smoke_checkpoint"; save_checkpoint(ck,params,opt_state,checkpoint_meta(cfg,{"hashes":{}},args.seed,"smoke",args.steps,params)); restored=load_checkpoint(ck)["params"]
    o1=model.apply({"params":params},batch,train=False)["rank"];o2=model.apply({"params":restored},batch,train=False)["rank"]; reload_match=bool(np.allclose(np.asarray(o1),np.asarray(o2),rtol=0,atol=0))
    # Cached CUDA truth identity survives feature order.
    cur_rows=current_rows(); cids=np.arange(min(len(cc["labels"]),args.limit)); cout=score(model,params,cc,cids,min(64,len(cids)))
    cm=metrics(cout["rank"],cc["labels"][cids],"smoke_current")
    guard=False
    try:
        from data import _read_result
        _read_result(MIN_GRU_OUT/"single_predicates_current140.b4rs",401,139)
    except ValueError: guard=True
    report={"environment":environment_info(),"config":asdict(cfg),"params":count,"param_range_ok":4_500_000<=count<=5_500_000,"same_seed_init":same,"forward_shapes":shapes,
            "initial_loss":float(loss),"losses":losses,"loss_decreased":losses[-1]<losses[0],"finite_gradients":all(np.isfinite(gnorms)),"gradient_norms":gnorms,"update_changed_params":before!=after,
            "params_finite":tree_all_finite(params),"checkpoint_reload_exact":reload_match,"probe_mismatch_guard":guard,"ranking_smoke":cm,
            "candidate_identity_example":cc["semantic_id"][0].decode(),"candidate_blob_sha_example":cc["blob_sha"][0].decode()}
    print(json.dumps(report,indent=2,sort_keys=True))
    if not (report["param_range_ok"] and same and reload_match and guard and report["finite_gradients"] and report["update_changed_params"] and report["params_finite"]): raise SystemExit(2)


def _full_manifest(): return json.loads((OUT/"features/manifest_full.json").read_text())


def _train_phase(args,kind,epochs,phase,init_ckpt=None):
    import jax
    cfg=cfg_for(args); cache=load_cache(kind,True); rows=history_rows()[0] if kind=="historical" else current_rows(); train_ids,val_ids=split_indices(kind,rows,args.split)
    batch0=batch_for(cache,np.asarray(train_ids[:min(args.batch_size,len(train_ids))])); model,params=init_model(cfg,batch0,args.seed)
    if init_ckpt: params=load_checkpoint(init_ckpt)["params"]
    labels=cache["labels"][train_ids]; weights=default_weights(labels,args); tx=make_tx(args.lr,args.weight_decay,args.clip); opt_state=tx.init(params)
    resume_rng=None; resume_updates=0
    if getattr(args,"resume_state",False) and init_ckpt:
        target={"params":params,"opt_state":opt_state,"rng":jax.random.PRNGKey(args.seed+100),"updates":jax.numpy.asarray(0,jax.numpy.int32)}
        restored=load_checkpoint(init_ckpt,target=target); params=restored["params"]; opt_state=restored["opt_state"]; resume_rng=restored["rng"]; resume_updates=int(np.asarray(restored["updates"]))
    numeric_keys=("tree_cat","tree_num","tree_mask","tree_parent","tree_depth","tree_rule","edit_cat","edit_num","edit_mask","case_features","case_mask","case_delta","heuristic","labels","group")
    device_data=to_jax({k:cache[k] for k in numeric_keys}); train_ids_jax=jax.numpy.asarray(train_ids,dtype=jax.numpy.int32); epoch_step=make_epoch_step(model,cfg,tx,weights,args.batch_size,args.effective_batch_size,len(train_ids))
    manifest=_full_manifest(); logdir=Path(args.out); logdir.mkdir(parents=True,exist_ok=True)
    other_kind="current" if kind=="historical" else "historical"; other_cache=load_cache(other_kind,True); other_rows=current_rows() if other_kind=="current" else history_rows()[0]; _,other_val=split_indices(other_kind,other_rows,args.split)
    def evaluate(tag,epoch):
        pred=score(model,params,cache,val_ids,args.eval_batch)["rank"]; own=metrics(pred,cache["labels"][val_ids],f"{kind}_val")
        opred=score(model,params,other_cache,other_val,args.eval_batch)["rank"]; other=metrics(opred,other_cache["labels"][other_val],f"{other_kind}_val")
        m={"tag":tag,"epoch":epoch,"params":count_params(params),kind:own,other_kind:other}; print("METRIC",json.dumps(m,sort_keys=True),flush=True); return m
    records=[]
    if phase=="historical":
        save_checkpoint(logdir/"untrained",params,None,checkpoint_meta(cfg,manifest,args.seed,phase,0,params,cli_config(args)));records.append(evaluate("untrained",0))
    rng=resume_rng if resume_rng is not None else jax.random.PRNGKey(args.seed+100); t0=time.perf_counter(); updates=resume_updates; updates_per_epoch=int(np.ceil(len(train_ids)/args.effective_batch_size))
    for ep in range(1,epochs+1):
        params,opt_state,rng,mean_loss,mean_gn=epoch_step(params,opt_state,device_data,train_ids_jax,rng); mean_loss.block_until_ready(); updates+=updates_per_epoch
        print(json.dumps({"phase":phase,"epoch":ep,"mean_loss":float(mean_loss),"mean_grad_norm":float(mean_gn),"updates":updates}),flush=True)
        if phase=="historical" and ep in (args.early_epoch,args.middle_epoch):
            tag="early" if ep==args.early_epoch else "middle"; save_checkpoint(logdir/tag,params,None,checkpoint_meta(cfg,manifest,args.seed,phase,ep,params,cli_config(args),updates));records.append(evaluate(tag,ep))
    tag="history_final" if kind=="historical" else "final"; save_checkpoint(logdir/tag,params,opt_state,checkpoint_meta(cfg,manifest,args.seed,phase,epochs,params,cli_config(args),updates),extra={"rng":rng,"updates":jax.numpy.asarray(updates,jax.numpy.int32)}); records.append(evaluate(tag,epochs))
    result={"phase":phase,"epochs":epochs,"updates":updates,"seconds":time.perf_counter()-t0,"records":records,"checkpoint":str((logdir/tag).resolve()),"weights":weights,"model_config":asdict(cfg),"run_config":cli_config(args),"environment":environment_info(),"dataset_hashes":manifest["hashes"]}
    (logdir/f"{phase}_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print("RESULT",json.dumps(result,sort_keys=True))


def cmd_train_hist(args):
    if args.dry_run: print(json.dumps({"command":"train-historical","config":vars(args),"schedule":"2 early / 6 middle / 10 history_final epochs, matching frozen MinGRU historical schedule"},default=str,indent=2)); return
    _train_phase(args,"historical",args.epochs,"historical")


def cmd_calibrate(args):
    if args.dry_run: print(json.dumps({"command":"calibrate","config":vars(args),"schedule":"20 current-domain epochs, frozen 319/81 split"},default=str,indent=2)); return
    _train_phase(args,"current",args.epochs,"calibration",args.checkpoint)


def cmd_evaluate(args):
    if args.dry_run: print(json.dumps({"command":"evaluate","config":vars(args)},default=str,indent=2)); return
    cfg=cfg_for(args); ck=load_checkpoint(args.checkpoint); cache=load_cache("current",True); rows=current_rows();tr,val=stratified_split(rows); batch=batch_for(cache,np.arange(min(args.batch_size,len(rows))));model,_=init_model(cfg,batch,args.seed); outs=score(model,ck["params"],cache,np.arange(len(rows)),args.batch_size)
    pred=outs["rank"]; neural=outs["rank_residual"]; labels=cache["labels"]; order=np.argsort(-pred,kind="mergesort"); ranks=np.empty(len(rows),int);ranks[order]=np.arange(1,len(rows)+1)
    records=[]
    for i,r in enumerate(rows): records.append({"index":i+1,"blob_sha":r.blob_sha,"semantic_id":r.semantic_id,"heldout":i in set(val),"score":float(pred[i]),"neural_score":float(neural[i]),"heuristic":float(cache["heuristic"][i]),"rank":int(ranks[i]),"safe":bool(labels[i,5]),"l2w":int(labels[i,6]),"w2l":int(labels[i,7]),"mean_delta":float(labels[i,8])})
    outp=Path(args.out);outp.parent.mkdir(parents=True,exist_ok=True);outp.write_text("\n".join(json.dumps(x,sort_keys=True) for x in records)+"\n")
    report={"current_holdout":metrics(pred[val],labels[val],"current_holdout"),"all400":metrics(pred,labels,"current400"),"safe_ranks":[x["rank"] for x in records if x["safe"]],"safe_records":[x for x in records if x["safe"]],"ranking_file":str(outp.resolve())}
    print(json.dumps(report,indent=2,sort_keys=True))


def proposer_metrics(order,labels,budget):
    ix=np.asarray(order[:budget]); safe=labels[:,5]>0; reg=labels[:,0]>0; md=labels[:,8]; l2w=labels[:,6]
    return {"budget":budget,"strict_safe_count":int(np.sum(safe[ix])),"loss_to_win_hit_rate":float(np.mean(l2w[ix]>0)),"any_win_regression_rate":float(np.mean(reg[ix])),"positive_margin_hit_rate":float(np.mean(md[ix]>0)),"mean_margin_delta":float(np.mean(md[ix])),"best_l2w":int(np.max(l2w[ix]))}


def cmd_compare(args):
    if args.dry_run: print(json.dumps({"command":"compare","config":vars(args)},default=str,indent=2)); return
    rows=[json.loads(x) for x in Path(args.rankings).read_text().splitlines() if x.strip()]; cache=load_cache("current",True); labels=cache["labels"]
    neural=np.argsort([-r["neural_score"] for r in rows],kind="mergesort"); residual=np.argsort([-r["score"] for r in rows],kind="mergesort"); heuristic=np.argsort([-r["heuristic"] for r in rows],kind="mergesort")
    rng=np.random.default_rng(args.seed); result={}
    for budget in (16,32,64,128):
        randoms=[proposer_metrics(rng.permutation(len(rows)),labels,budget) for _ in range(args.random_repeats)]
        rmean={k:float(np.mean([x[k] for x in randoms])) for k in randoms[0] if k!="budget"};rmean["budget"]=budget
        result[str(budget)]={"random_mean":rmean,"heuristic":proposer_metrics(heuristic,labels,budget),"transformer_neural":proposer_metrics(neural,labels,budget),"transformer_residual":proposer_metrics(residual,labels,budget)}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True))


def cmd_benchmark(args):
    import jax
    cfg=cfg_for(args); hc,_=smoke_cache(cfg,max(args.batch_size,128)); cache=hc; ids=np.arange(args.batch_size);batch=batch_for(cache,ids);model,params=init_model(cfg,batch,args.seed);weights=default_weights(cache["labels"][ids]);tx=make_tx(args.lr,args.weight_decay,args.clip);opt=tx.init(params);step=make_train_step(model,cfg,tx,weights);key=jax.random.PRNGKey(args.seed+9)
    t=time.perf_counter();key,sk=jax.random.split(key);params,opt,l,aux,gn=step(params,opt,batch,sk);l.block_until_ready();jit_train=time.perf_counter()-t
    for _ in range(args.warmup): key,sk=jax.random.split(key);params,opt,l,aux,gn=step(params,opt,batch,sk);l.block_until_ready()
    t=time.perf_counter()
    for _ in range(args.steps): key,sk=jax.random.split(key);params,opt,l,aux,gn=step(params,opt,batch,sk);l.block_until_ready()
    train_s=time.perf_counter()-t
    f=jax.jit(lambda p,b:model.apply({"params":p},b,train=False)["rank"]);t=time.perf_counter();y=f(params,batch);y.block_until_ready();jit_inf=time.perf_counter()-t
    for _ in range(args.warmup): y=f(params,batch);y.block_until_ready()
    t=time.perf_counter()
    for _ in range(args.steps*4): y=f(params,batch);y.block_until_ready()
    inf_s=time.perf_counter()-t
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.used","--format=csv,noheader,nounits"],text=True).strip(); mem=jax.devices()[0].memory_stats() or {}
    print(json.dumps({"config":asdict(cfg),"params":count_params(params),"batch_size":args.batch_size,"tree_tokens":cfg.tree_cap,"edit_tokens":cfg.edit_cap,"case_tokens":cfg.case_cap,"precision":cfg.compute_dtype,
                      "train_jit_seconds":jit_train,"train_examples_s":args.batch_size*args.steps/train_s,"inference_jit_seconds":jit_inf,"inference_candidates_s":args.batch_size*(args.steps*4)/inf_s,"nvidia_smi_util_mem":gpu,
                      "jax_peak_bytes_in_use":int(mem.get("peak_bytes_in_use",0)),"jax_bytes_limit":int(mem.get("bytes_limit",0))},indent=2,sort_keys=True))


def cmd_resume(args):
    if args.dry_run: print(json.dumps({"command":"resume","checkpoint":args.checkpoint,"epochs":args.epochs,"kind":args.kind},indent=2)); return
    args.resume_state=True; _train_phase(args,args.kind,args.epochs,"resume",args.checkpoint)


def common(p):
    p.add_argument("--seed",type=int,default=73);p.add_argument("--attention-impl",choices=("xla","cudnn"),default="xla");p.add_argument("--ablation",choices=("complete","t0","t1","t2","t3","t4","t5"),default="complete")


def train_args(p):
    common(p);p.add_argument("--batch-size",type=int,default=32,help="GPU microbatch");p.add_argument("--effective-batch-size",type=int,default=1024,help="optimizer batch via gradient accumulation");p.add_argument("--eval-batch",type=int,default=128);p.add_argument("--lr",type=float,default=3e-4);p.add_argument("--weight-decay",type=float,default=1e-4);p.add_argument("--clip",type=float,default=2.0);p.add_argument("--split",choices=("primary","secondary"),default="primary");
    p.add_argument("--w-reg",type=float,default=6.0);p.add_argument("--w-flip",type=float,default=4.0);p.add_argument("--w-margin",type=float,default=1.0);p.add_argument("--w-worst",type=float,default=3.0);p.add_argument("--w-rank",type=float,default=6.0);p.add_argument("--w-case",type=float,default=1.5)
    p.add_argument("--out",default=str(OUT/"runs/editformer5m"));p.add_argument("--dry-run",action="store_true")


def parser():
    ap=argparse.ArgumentParser(description="BonsaiEditFormer-5M structural BKG4 edit ranker");sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("stack-check");common(p);p.set_defaults(fn=cmd_stack)
    p=sp.add_parser("prepare-data");common(p);p.add_argument("--limit",type=int);p.add_argument("--force",action="store_true");p.set_defaults(fn=cmd_prepare)
    p=sp.add_parser("smoke");common(p);p.add_argument("--limit",type=int,default=128);p.add_argument("--batch-size",type=int,default=32);p.add_argument("--steps",type=int,default=8);p.add_argument("--lr",type=float,default=3e-4);p.add_argument("--weight-decay",type=float,default=1e-4);p.add_argument("--clip",type=float,default=2.0);p.set_defaults(fn=cmd_smoke)
    p=sp.add_parser("train-historical");train_args(p);p.add_argument("--epochs",type=int,default=10);p.add_argument("--early-epoch",type=int,default=2);p.add_argument("--middle-epoch",type=int,default=6);p.set_defaults(fn=cmd_train_hist)
    p=sp.add_parser("calibrate");train_args(p);p.set_defaults(lr=1e-4,effective_batch_size=128);p.add_argument("--epochs",type=int,default=20);p.add_argument("--checkpoint",required=True);p.set_defaults(fn=cmd_calibrate)
    p=sp.add_parser("evaluate");common(p);p.add_argument("--checkpoint",required=True);p.add_argument("--batch-size",type=int,default=128);p.add_argument("--out",default=str(OUT/"runs/editformer5m/current_rankings.jsonl"));p.add_argument("--dry-run",action="store_true");p.set_defaults(fn=cmd_evaluate)
    p=sp.add_parser("compare");p.add_argument("--rankings",default=str(OUT/"runs/editformer5m/current_rankings.jsonl"));p.add_argument("--out",default=str(OUT/"runs/editformer5m/search_comparison.json"));p.add_argument("--seed",type=int,default=123);p.add_argument("--random-repeats",type=int,default=20);p.add_argument("--dry-run",action="store_true");p.set_defaults(fn=cmd_compare)
    p=sp.add_parser("benchmark");common(p);p.add_argument("--batch-size",type=int,default=32);p.add_argument("--warmup",type=int,default=2);p.add_argument("--steps",type=int,default=10);p.add_argument("--lr",type=float,default=3e-4);p.add_argument("--weight-decay",type=float,default=1e-4);p.add_argument("--clip",type=float,default=2.0);p.set_defaults(fn=cmd_benchmark)
    p=sp.add_parser("resume");train_args(p);p.add_argument("--checkpoint",required=True);p.add_argument("--kind",choices=("historical","current"),default="historical");p.add_argument("--epochs",type=int,default=1);p.set_defaults(fn=cmd_resume)
    return ap


def main():
    args=parser().parse_args(); return args.fn(args)

if __name__=="__main__": main()
