import importlib.util, pathlib, sys
HERE=pathlib.Path(__file__).resolve().parent

def _load_shadow():
    d=HERE/"q8_shadow"
    names={p.stem for p in d.glob("*.py")}
    for n in names:
        sys.modules.pop(n,None)
    sys.modules.pop("runtime_gateway",None)
    sys.path.insert(0,str(d))
    try:
        spec=importlib.util.spec_from_file_location("q8_shadow_entry",d/"main.py")
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        fn=mod.agent
    finally:
        sys.path.remove(str(d))
    return fn

_Q8=_load_shadow()
# Force the primary candidate to receive fresh sibling modules rather than the shadow cache.
for _p in HERE.glob("*.py"):
    if _p.stem!="main": sys.modules.pop(_p.stem,None)
sys.modules.pop("runtime_gateway",None)
from meta_gold_v29_minimax import agent as _META
_USE_Q8={0:False,1:False}
_WB_SKIP={0:False,1:False}
_WOOL_DEBT={0:0,1:0}
_STRAW_OPT_DEBT={0:0,1:0}
_W14T={0:False,1:False}
_STRAW_DEBT={0:{},1:{}}
_LATE_STRAW={0:0,1:0}
_WDEBT={0:0,1:0}
_MILK_DUE={0:{},1:{}}

def agent(observation):
    seat=int(observation.get("player",0) or 0); step=int(observation.get("step",0) or 0)
    if step==0: _USE_Q8[seat]=False; _W14T[seat]=False; _WB_SKIP[seat]=False; _WOOL_DEBT[seat]=0; _STRAW_DEBT[seat]={}; _LATE_STRAW[seat]=0; _WDEBT[seat]=0; _MILK_DUE[seat]={}
    qa=_Q8(observation)
    ma=_META(observation)
    if step==1:
        farms=list(observation.get("farms") or [])
        if len(farms)>1:
            opp=farms[1-seat]
            _W14T[seat]=(len(opp.get("hands",[]) or [])==5 and float(opp.get("money",0) or 0)==0.0)
            _USE_Q8[seat]=((len(opp.get("hands",[]) or [])==5 and float(opp.get("money",0) or 0) in (17.0,20.0)) or (len(opp.get("hands",[]) or [])>=6 and float(opp.get("money",0) or 0)>=600.0))
    if _W14T.get(seat,False):
        import copy as _cp
        _rules={576:(1,586),600:(1,604),624:(40,640)}
        if step in _rules:
            th,due=_rules[step]; price=int((observation.get("market",{}).get("prices",{}) or {}).get("STRAWBERRY",0) or 0)
            planned=sum(max(0,int(o[2])) for o in (ma.get("market") or []) if len(o)>=3 and o[0]=="SELL" and o[1]=="STRAWBERRY")
            if planned>0 and price<=th:
                ma=_cp.deepcopy(ma); ma["market"]=[list(o) for o in (ma.get("market") or []) if not (len(o)>=2 and o[0]=="SELL" and o[1]=="STRAWBERRY")]; _STRAW_DEBT[seat][due]=_STRAW_DEBT[seat].get(due,0)+planned
        dueq=int(_STRAW_DEBT[seat].pop(step,0) or 0)
        if dueq>0 and len(ma.get("market") or [])<10:
            ma=_cp.deepcopy(ma); ma["market"]=list(ma.get("market") or [])+[["SELL","STRAWBERRY",dueq]]
        if step>=648:
            prod="STRAWBERRY"; price=int((observation.get("market",{}).get("prices",{}) or {}).get(prod,0) or 0); stock=int(((observation.get("private",{}).get("shed",{}) or {}).get(prod,0) or 0))
            orders=[list(o) for o in (ma.get("market") or [])]; qplan=sum(max(0,int(o[2])) for o in orders if len(o)>=3 and o[0]=="SELL" and o[1]==prod); held=min(int(_LATE_STRAW.get(seat,0) or 0),stock)
            if held>0 and (price>=60 or step>=700):
                out=[o for o in orders if not (len(o)>=2 and o[0]=="SELL" and o[1]==prod)]; total=min(stock,held+qplan)
                if total>0 and len(out)<10: out.append(["SELL",prod,total]); _LATE_STRAW[seat]=0; ma=_cp.deepcopy(ma); ma["market"]=out
            elif held>0:
                add=min(qplan,max(0,stock-held)); _LATE_STRAW[seat]=held+add
                if qplan>0: ma=_cp.deepcopy(ma); ma["market"]=[o for o in orders if not (len(o)>=2 and o[0]=="SELL" and o[1]==prod)]
            elif qplan>0 and price<=40 and step<700:
                q=min(qplan,stock)
                if q>0: _LATE_STRAW[seat]=q; ma=_cp.deepcopy(ma); ma["market"]=[o for o in orders if not (len(o)>=2 and o[0]=="SELL" and o[1]==prod)]
    if _W14T.get(seat,False):
        import copy as _tc
        if step==361:
            ma=_tc.deepcopy(ma); ma["market"]=[list(o) for o in (ma.get("market") or []) if not (len(o)>=3 and o[0]=="BUY_ANIMAL" and o[1]=="COW")]
        if step==505:
            milkp=int((observation.get("market",{}).get("prices",{}) or {}).get("MILK",0) or 0)
            if milkp>50 and len(ma.get("market") or [])<10:
                ma=_tc.deepcopy(ma); ma["market"]=list(ma.get("market") or [])+[["BUY_ANIMAL","COW",1]]
        # WOOL: suppress the canonical step-600 floor dump and release only after observable recovery.
        if step==600:
            wp=int((observation.get("market",{}).get("prices",{}) or {}).get("WOOL",0) or 0); orders=[list(o) for o in (ma.get("market") or [])]; q=sum(max(0,int(o[2])) for o in orders if len(o)>=3 and o[0]=="SELL" and o[1]=="WOOL")
            if q>0 and wp==1:
                _WDEBT[seat]=q; ma=_tc.deepcopy(ma); ma["market"]=[o for o in orders if not (len(o)>=2 and o[0]=="SELL" and o[1]=="WOOL")]
        if _WDEBT.get(seat,0)>0 and 601<=step<=668:
            wp=int((observation.get("market",{}).get("prices",{}) or {}).get("WOOL",0) or 0)
            if wp>=100 or step==668:
                stock=int(((observation.get("private",{}).get("shed",{}) or {}).get("WOOL",0) or 0)); q=min(stock,int(_WDEBT[seat]))
                if q>0 and len(ma.get("market") or [])<10: ma=_tc.deepcopy(ma); ma["market"]=list(ma.get("market") or [])+[["SELL","WOOL",q]]
                _WDEBT[seat]=0
        # MILK: bounded low-quote tranches, each released before the next production block.
        m_rules={648:(15,660),670:(10,684),672:(10,684),690:(15,713),691:(15,713)}
        if step in m_rules:
            th,due=m_rules[step]; mp=int((observation.get("market",{}).get("prices",{}) or {}).get("MILK",0) or 0); orders=[list(o) for o in (ma.get("market") or [])]; q=sum(max(0,int(o[2])) for o in orders if len(o)>=3 and o[0]=="SELL" and o[1]=="MILK")
            if q>0 and mp<=th:
                _MILK_DUE[seat][due]=_MILK_DUE[seat].get(due,0)+q; ma=_tc.deepcopy(ma); ma["market"]=[o for o in orders if not (len(o)>=2 and o[0]=="SELL" and o[1]=="MILK")]
        mq=int(_MILK_DUE[seat].pop(step,0) or 0)
        if mq>0:
            stock=int(((observation.get("private",{}).get("shed",{}) or {}).get("MILK",0) or 0)); q=min(stock,mq)
            if q>0 and len(ma.get("market") or [])<10: ma=_tc.deepcopy(ma); ma["market"]=list(ma.get("market") or [])+[["SELL","MILK",q]]
    if _W14T.get(seat,False):
        if step==576:
            _WB_SKIP[seat]=int((observation.get("market",{}).get("prices",{}) or {}).get("WHEAT",0) or 0)<=46
        if _WB_SKIP.get(seat,False) and step>=576:
            orders=[list(o) for o in (ma.get("market") or [])]
            nm=[o for o in orders if not (len(o)>=2 and o[0]=="BUY_PRODUCT" and o[1]=="WHEAT")]
            if len(nm)!=len(orders):
                import copy as _wbcp
                ma=_wbcp.deepcopy(ma); ma["market"]=nm
    if _W14T.get(seat,False) and 240<=step<=718 and "YARN_STORE" in list((observation.get("town",{}) or {}).get("unlocked_shops",[]) or []):
        import copy as _wc, math as _wm
        orders=[list(o) for o in (ma.get("market") or [])]
        qplan=sum(max(0,int(o[2])) for o in orders if len(o)>=3 and o[0]=="SELL" and o[1]=="WOOL")
        old=max(0,int(_WOOL_DEBT.get(seat,0) or 0))
        priv=observation.get("private",{}) or {}
        stock=max(0,int((priv.get("shed",{}) or {}).get("WOOL",0) or 0))
        invs=list(priv.get("inventories",[]) or [])
        cmds=[ma.get("farmer") or ["PASS"]]+list(ma.get("hands") or [])
        drops=0
        for iv,cmd in zip(invs,cmds):
            if isinstance(iv,dict) and isinstance(cmd,list) and cmd and cmd[0]=="DROP": drops+=max(0,int(iv.get("WOOL",0) or 0))
        due=min(stock+drops,old+qplan)
        if due>0:
            mi=max(0,int((observation.get("market",{}).get("inventory",{}) or {}).get("WOOL",10000) or 10000))
            def wp(x):
                if x<10000:
                    return max(1,int(round(200.0+(40.0/_wm.log(106.0))*_wm.log(1.0+10000-x))))
                return max(1,int(round(200.0-(640.0/11025.0)*((x-10000.0)**2))))
            sellq=due if step==718 else 0
            if step<718:
                for k in range(due):
                    if wp(mi+k)>=70: sellq+=1
                    else: break
            # v16 minimax reserve bound: never carry more than 16 WOOL as controller debt.
            sellq=max(sellq,max(0,due-16))
            nm=[o for o in orders if not (len(o)>=2 and o[0]=="SELL" and o[1]=="WOOL")]
            if sellq>0 and len(nm)<10: nm.append(["SELL","WOOL",sellq])
            if qplan>0 or old>0:
                ma=_wc.deepcopy(ma); ma["market"]=nm
            _WOOL_DEBT[seat]=max(0,due-sellq)
    # v18 post-unlock STRAWBERRY marginal controller: all 8 town shops are known by step576.
    # Bound reserve to 16 units so this cannot create v15-style shed crowding.
    _straw_demand=sum(1 for _s in list((observation.get("town",{}) or {}).get("unlocked_shops",[]) or []) if _s in ("BRUNCH_SPOT","ICE_CREAM_SHOP","SMOOTHIE_SHOP","FARMERS_MARKET"))
    if _W14T.get(seat,False) and 576<=step<=718 and _straw_demand>=4:
        import copy as _sc, math as _sm
        _orders=[list(_o) for _o in (ma.get("market") or [])]
        _qplan=sum(max(0,int(_o[2])) for _o in _orders if len(_o)>=3 and _o[0]=="SELL" and _o[1]=="STRAWBERRY")
        _old=max(0,int(_STRAW_OPT_DEBT.get(seat,0) or 0))
        _priv=observation.get("private",{}) or {}; _stock=max(0,int((_priv.get("shed",{}) or {}).get("STRAWBERRY",0) or 0))
        _invs=list(_priv.get("inventories",[]) or []); _cmds=[ma.get("farmer") or ["PASS"]]+list(ma.get("hands") or [])
        _drops=0
        for _iv,_cmd in zip(_invs,_cmds):
            if isinstance(_iv,dict) and isinstance(_cmd,list) and _cmd and _cmd[0]=="DROP": _drops+=max(0,int(_iv.get("STRAWBERRY",0) or 0))
        _due=min(_stock+_drops,_old+_qplan)
        if _due>0:
            _mi=max(0,int((observation.get("market",{}).get("inventory",{}) or {}).get("STRAWBERRY",10000) or 10000))
            def _sp(_x):
                if _x<10000:
                    return max(1,int(round(120.0+8.4*_sm.sqrt(10000.0-_x))))
                return max(1,int(round(120.0-1.92*(_x-10000.0))))
            _sell=_due if step==718 else 0
            if step<718:
                for _k in range(_due):
                    if _sp(_mi+_k)>=120: _sell+=1
                    else: break
                _sell=max(_sell,max(0,_due-16))
            _nm=[_o for _o in _orders if not (len(_o)>=2 and _o[0]=="SELL" and _o[1]=="STRAWBERRY")]
            if _sell>0 and len(_nm)<10: _nm.append(["SELL","STRAWBERRY",_sell])
            if _qplan>0 or _old>0:
                ma=_sc.deepcopy(ma); ma["market"]=_nm
            _STRAW_OPT_DEBT[seat]=max(0,_due-_sell)
    return qa if _USE_Q8.get(seat,False) else ma
