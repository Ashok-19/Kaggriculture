import importlib.util, pathlib

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('meta_gold_v10_base', HERE / 'meta_gold_v8.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
rg = base.rg

_DEP_CROPS = {'WHEAT', 'STRAWBERRY', 'MELON'}


def _is_dep(a):
    if not isinstance(a, list) or not a:
        return False
    if a[0] == 'BUILD_PASTURE':
        return True
    return len(a) >= 2 and a[0] == 'PLANT' and a[1] in _DEP_CROPS


def _trace(route, source_step, actor):
    return list(rg._trace_actor_action(route, source_step, actor) or ['PASS'])


def _hybrid_weed_repair(obs, action, route, step, state):
    action = rg._align(rg._copy_action(action), obs)
    farm = rg._farm(obs, int(rg._get(obs, 'player', 0) or 0))
    positions = [rg._get(farm, 'farmer'), *list(rg._get(farm, 'hands', []) or [])]
    unit = [action.get('farmer', ['PASS']), *list(action.get('hands') or [])]
    active = state['weed']
    if step % 24 == 0:
        active.clear()

    for actor, tx in list(active.items()):
        idx = 0 if actor == 'farmer' else int(actor) + 1
        if idx >= len(unit):
            active.pop(actor, None)
            continue
        position = positions[idx] if idx < len(positions) else None
        tile = rg._tile_at(farm, position)

        # Nested mode is entered only if an already-delayed dependency reaches a
        # second WEED.  From there, a canonical cursor keeps the blocked action
        # pending and consumes due PASS slots as catch-up slack.
        if 'nested_cursor' in tx:
            cursor = int(tx['nested_cursor'])
            if cursor < 0 or cursor >= len(route) or cursor > step:
                active.pop(actor, None)
                continue
            intended = _trace(route, cursor, actor)
            if _is_dep(intended) and isinstance(tile, dict) and tile.get('kind') == 'WEED':
                unit[idx] = ['DIG']
                continue
            unit[idx] = intended
            cursor += 1
            while cursor <= step and cursor < len(route) and _trace(route, cursor, actor) == ['PASS']:
                cursor += 1
            if cursor > step:
                active.pop(actor, None)
            else:
                tx['nested_cursor'] = cursor
            continue

        age = step - int(tx['start'])
        original = list(tx['intended'])
        dep = _is_dep(original)
        if age == 1:
            unit[idx] = original
        elif dep and age >= 2:
            replay = _trace(route, step - 1, actor)
            if _is_dep(replay) and isinstance(tile, dict) and tile.get('kind') == 'WEED':
                unit[idx] = ['DIG']
                tx['nested_cursor'] = step - 1
                continue
            unit[idx] = replay
            if _trace(route, step, actor) == ['PASS'] or age > 24:
                active.pop(actor, None)
        elif 2 <= age <= 1 + rg.WEED_REPLAY_STEPS:
            unit[idx] = _trace(route, step - 1, actor)
        else:
            active.pop(actor, None)

    for idx, (position, intended) in enumerate(zip(positions, unit)):
        actor = 'farmer' if idx == 0 else idx - 1
        if actor in active or not isinstance(intended, list) or not intended or intended[0] not in ('BUILD_PASTURE', 'PLANT'):
            continue
        tile = rg._tile_at(farm, position)
        if isinstance(tile, dict) and tile.get('kind') == 'WEED':
            active[actor] = {'start': step, 'intended': list(intended)}
            unit[idx] = ['DIG']

    action['farmer'] = unit[0] if unit else ['PASS']
    action['hands'] = unit[1:]
    return rg._align(action, obs)


rg._weed_repair = _hybrid_weed_repair
agent = base.agent
