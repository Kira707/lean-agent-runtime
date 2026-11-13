"""Formal metrics and a reproducible manual-audit order; no extractor in sampling."""
import json
from pathlib import Path
import random

from eval.metrics import summarize, flips
from eval.report import audit_runs
from eval.runner import dump
from eval.fact_audit import fact_totals


def report_f1(folder):
    folder=Path(folder)
    rows=json.loads((folder/'runs.json').read_text())
    protocol=json.loads((folder/'protocol.json').read_text())
    metrics=summarize(rows,protocol['jobs'])
    _,audits,_=audit_runs(folder,rows)
    by_id={r['run_id']:r for r in rows}
    compliance={}
    for variant in metrics:
        group=[a for a in audits if by_id[a['run_id']]['variant']==variant]
        errors=[(a,e) for a in group for e in a['numeric_facts']
                if e['verdict']=='violation' and e['user_visible']]
        compliance[variant]={'error_runs':len({a['run_id'] for a,e in errors}),
            'error_events':len(errors),
            'blocked_messages':sum(a['harness_audit']['blocked_messages'] for a in group),
            'false_interceptions':sum(a['harness_audit']['false_interceptions'] for a in group),
            'unknown_interceptions':sum(a['harness_audit']['unknown_interceptions'] for a in group),
            'extra_http_attempts':sum(a['harness_audit']['extra_http_attempts'] for a in group),
            'extra_steps':sum((by_id[a['run_id']].get('harness') or {}).get('extra_steps',0) for a in group)}
        compliance[variant]['retail_facts']=fact_totals(group)
        compliance[variant]['false_interception_examples']=[
            {'run_id':a['run_id'],'trace_path':by_id[a['run_id']]['trace_path'],**case}
            for a in group for case in a['harness_audit']['interceptions'] if case['verdict']=='false']
    complete=all(m['complete'] for m in metrics.values())
    changed={v:flips(metrics['baseline']['per_task_successes'],metrics[v]['per_task_successes'])
             for v in metrics if v!='baseline'} if complete else {}
    dump(folder/'summary.json',{'metrics':metrics,'compliance':compliance,'task_flips':changed,
                              'manual_audit_required':True,'manual_audit_sample_size':10})
    lines=['# F1结果','','| 组别 | 完成 | pass¹ | pass³ | 已发错误轨迹／事件 |',
           '|---|---:|---:|---:|---:|']
    for variant in metrics:
        m=metrics[variant];c=compliance[variant]
        p1=f"{m['pass1']:.2%}" if complete else '未完成'
        p3=f"{m['pass3']:.2%}" if complete else '未完成'
        lines.append(f"| {variant} | {m['completed']}/{m['planned']} | {p1} | {p3} | {c['error_runs']}／{c['error_events']} |")
    lines+=['','## 任务层面翻转明细','']
    for variant,changes in changed.items():
        lines += [f"- {variant}｜{r['task']}：{r['before']}/3 → {r['after']}/3" for r in changes]
    if 'all_guards' in compliance:
        combined=compliance['all_guards']
        lines+=['','## H1+H2误拦','',
            f"已确认误拦：{combined['false_interceptions']}次；待判：{combined['unknown_interceptions']}次。",
            '按被拦消息计数；同条消息H1与H2同时触发仅计一次。误拦表示该消息的全部拦截理由均不成立，未判定不当作零误拦。']
        examples=combined['false_interception_examples']
        if not examples:
            lines.append('当前无已确认的误拦样例；待判数见上。')
        for case in examples[:3]:
            lines += [f"- {case['run_id']}，M{case['message_index']}，触发：{', '.join(case['mechanisms'])}。",
                '', '> '+case['text'].replace('\n','\n> '), '',
                f"  判定依据：{(case.get('review') or {}).get('reason','见trace中的原始事实及审查记录')}",
                f"  [原始轨迹]({case['trace_path']})"]
        lines.append('全部已确认误拦样例保存在summary.json的compliance.all_guards.false_interception_examples。')
    lines+=['','## 独立人工核对','',
        'H与评测共用提取器，须沿independent_audit_order.json的随机顺序直接阅读H组对话，选满10条实际提到银行卡号／付款标识的轨迹。不调用提取器作筛选或核对，记录跳过的无相关表述轨迹；核对结果随后附在本报告。']
    (folder/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if complete and not (folder/'independent_audit_order.json').exists():
        audit_variant='all_guards' if 'all_guards' in metrics else 'payment_guard'
        candidates=sorted(r['run_id'] for r in rows if r['variant']==audit_variant)
        seed=protocol['core']['audit']['seed'];random.Random(seed).shuffle(candidates)
        dump(folder/'independent_audit_order.json',{'seed':seed,'required':10,'variant':audit_variant,
             'extractor_used_for_selection':False,'candidate_order':candidates})
    return metrics,compliance
