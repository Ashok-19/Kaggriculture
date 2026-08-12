import importlib.util, pathlib

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('meta_gold_v8_base', HERE / 'meta_gold_v1.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
rg = base.rg
_ORIG_ROUTE = base._route
_ORIG_EVAC = rg._apply_dayend_evacuation
SECONDARY_PRODUCTS = tuple(rg.EVAC_PRODUCTS) + ('FERTILIZER',)


def _first2(obs):
    return list((obs.get('town', {}) or {}).get('unlocked_shops', []) or [])[:2]


def _blocked(first2):
    return any(s in ('SMOOTHIE_SHOP', 'PIZZA_SHOP') for s in first2)


def _route(obs, branch):
    step = int(obs.get('step', 0))
    if branch == 'kaito' and step >= 20:
        first2 = _first2(obs)
        yarn = 'YARN_STORE' in first2
        blocked = _blocked(first2)
        if step >= 144 and (not yarn or blocked):
            return base.CURRENT
        if step >= 192 and first2 and first2[0] == 'YARN_STORE':
            return base.YARN
        return base.CORE
    return _ORIG_ROUTE(obs, branch)


base._route = _route


def _use_secondary(obs, seat):
    if base._BRANCH.get(seat) != 'kaito':
        return False
    first2 = _first2(obs)
    if not first2 or 'YARN_STORE' not in first2:
        return False
    if _blocked(first2):
        return True
    return first2[0] == 'YARN_STORE' and len(first2) >= 2 and first2[1] in ('YARN_STORE', 'BAKERY')


def _secondary_evac(action, obs, step, state, route):
    out = _ORIG_EVAC(action, obs, step, state, route)
    key = 'strict_secondary_evac_v8'
    seat = int(obs.get('player', 0))
    if not _use_secondary(obs, seat):
        state[key] = []
        return out
    if step % 24 == 0:
        state[key] = []
    remaining = 24 - (step % 24)
    if remaining <= 0 or remaining > rg.EVAC_PASS_HORIZON:
        return out

    farm = rg._farm(obs, seat)
    positions = rg._unit_positions(farm)
    invs = rg._private_inventories(obs)
    plans = list(state.get(key) or [])
    if not plans:
        primary = None
        ep = state.get('evac_plan')
        if isinstance(ep, dict):
            primary = ep.get('actor')
        for idx, (position, inventory) in enumerate(zip(positions, invs)):
            if idx == primary or not position or not isinstance(inventory, dict):
                continue
            live = out.get('farmer') if idx == 0 else (
                (out.get('hands') or [])[idx - 1]
                if idx - 1 < len(out.get('hands') or []) else ['PASS']
            )
            if list(live or ['PASS']) != rg._actor_action(route, step, idx):
                continue
            end = min(len(route), step + remaining)
            future = [rg._actor_action(route, t, idx) for t in range(step, end)]
            if any(a != ['PASS'] for a in future):
                continue
            if not any(max(0, int(inventory.get(p, 0) or 0)) > 0 for p in SECONDARY_PRODUCTS):
                continue
            target = min(rg.SHED_ACCESS, key=lambda q: rg._distance(position, q))
            if rg._distance(position, target) + 1 > remaining:
                continue
            plans.append({'actor': idx, 'target': target})
        state[key] = plans

    if not plans:
        return out

    shed = (obs.get('private', {}) or {}).get('shed', {}) or {}
    free = max(0, 100 - sum(max(0, int(v or 0)) for v in shed.values()))
    market = [list(o) for o in (out.get('market') or [])]
    keep = []
    for plan in plans:
        idx = int(plan['actor'])
        if idx >= len(positions):
            continue
        live = out.get('farmer') if idx == 0 else (
            (out.get('hands') or [])[idx - 1]
            if idx - 1 < len(out.get('hands') or []) else ['PASS']
        )
        if list(live or ['PASS']) != rg._actor_action(route, step, idx):
            continue
        position = positions[idx]
        target = plan['target']
        inventory = invs[idx] if idx < len(invs) and isinstance(invs[idx], dict) else {}
        if tuple(position) != tuple(target):
            rg._set_actor_action(out, idx, rg._step_toward(position, target))
            keep.append(plan)
            continue
        if free <= 0 or len(market) >= 10:
            keep.append(plan)
            continue
        best = None
        for product in SECONDARY_PRODUCTS:
            qty = max(0, int(inventory.get(product, 0) or 0))
            if qty <= 0:
                continue
            score = qty * rg.EVAC_BASE_VALUE.get(product, 1)
            if best is None or score > best[0]:
                best = (score, product, qty)
        if best is None:
            continue
        _, product, qty = best
        qty = min(qty, free)
        if qty <= 0:
            keep.append(plan)
            continue
        rg._set_actor_action(out, idx, ['PLACE', product, qty])
        market.append(['SELL', product, qty])
        free -= qty
        if any(p != product and max(0, int(inventory.get(p, 0) or 0)) > 0 for p in SECONDARY_PRODUCTS):
            keep.append(plan)

    state[key] = keep
    out['market'] = market
    return rg._align(out, obs)


rg._apply_dayend_evacuation = _secondary_evac
agent = base.agent
