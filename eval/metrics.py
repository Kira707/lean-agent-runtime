from collections import defaultdict
from math import comb, ceil
from statistics import median


def summarize(rows, jobs):
    expected = {j['run_id'] for j in jobs}
    if len(expected) != len(jobs) or len({r['run_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate planned or recorded run_id')
    complete = {r['run_id'] for r in rows} == expected and all(
        r['status'] not in ('not_started','running') for r in rows)
    groups=defaultdict(list)
    for row in rows: groups[(row['variant'],row['task'])].append(row['reward']==1)
    variants={j['variant'] for j in jobs}
    output={}
    for variant in sorted(variants):
        current=[r for r in rows if r['variant']==variant]
        grouped={task:g for (v,task),g in groups.items() if v==variant}
        result={'complete':complete,'planned':sum(j['variant']==variant for j in jobs),
            'completed':sum(r['status'] not in ('not_started','running') for r in current),
            'successes':sum(r['reward']==1 for r in current),
            'per_task_successes':{t:sum(g) for t,g in grouped.items()},
            'pass1':None,'pass2':None,'pass3':None}
        if complete:
            for k in (1,2,3):
                result[f'pass{k}']=sum(comb(sum(g),k)/comb(len(g),k) for g in grouped.values())/len(grouped)
        costs=[r.get('usage',{}).get('estimated_cost_cny') for r in current]
        result['estimated_cost_cny']=sum(costs) if all(c is not None for c in costs) else None
        attempted=[r for r in current if r['status'] not in ('not_started','running')]
        walls=sorted(r['wall_ms'] for r in attempted if r.get('wall_ms') is not None)
        result.update(mean_steps=sum(r.get('steps',0) for r in attempted)/len(attempted) if attempted else None,
            budget_exceeded_runs=sum(r['status']=='budget_exceeded' for r in attempted),
            p50_wall_ms=median(walls) if walls else None,
            p95_wall_ms=walls[ceil(.95*len(walls))-1] if walls else None)
        result['wall_sample_count']=len(walls)
        output[variant]=result
    return output


def flips(before, after):
    if set(before)!=set(after): raise ValueError('Paired task sets differ')
    return [{'task':task,'before':before[task],'after':after[task]}
            for task in sorted(before) if before[task]!=after[task]]
