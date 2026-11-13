import json
from pathlib import Path
from eval.metrics import summarize, flips
from eval.divergence import first_unmatched
from eval.runner import dump
from scripts.audit_compliance import scan_trace
from scripts.numeric_facts import scan_numeric, verify_card_claims
from eval.payment_audit import audit_interceptions
from eval.fact_audit import audit_facts, fact_totals
from m0.protocol import git

ROOT=Path(__file__).resolve().parents[1]


def audit_runs(folder, rows):
    catalog=json.loads((ROOT/'vendor/tau-bench/tau_bench/envs/retail/data/users.json').read_text())
    users=set(catalog)
    audits=[]
    failures=[]
    review_path=folder/'fact_block_review.json'
    block_reviews=json.loads(review_path.read_text()) if review_path.exists() else {}
    for row in rows:
        if row['status'] in ('running','not_started'): continue
        path=folder/row['run_id']
        messages=json.loads((path/'messages.json').read_text())
        events=[json.loads(l) for l in (path/'trace.jsonl').read_text().splitlines()]
        replies=[e['response']['choices'][0]['message'] for e in events
                 if e['event']=='model_call' and e.get('actor')=='agent' and e.get('response')]
        # Failed/budgeted runs may have an unconsumed model response. Compare the
        # consumed prefix, retaining all raw events separately in trace.jsonl.
        consumed=sum(m['role']=='assistant' for m in messages)
        audits.append({'run_id':row['run_id'],'task':row['task'],'reward':row['reward'],
            **scan_trace(messages,replies[:consumed],users),'numeric_facts':scan_numeric(messages)})
        auth=audits[-1]['auth_events']
        methods=catalog[auth[0]['user_id']]['payment_methods'] if auth else {}
        verify_card_claims(audits[-1]['numeric_facts'],methods)
        audits[-1]['harness_audit']=audit_interceptions(audits[-1]['numeric_facts'],events,methods,block_reviews.get(row['run_id']))
        audits[-1]['retail_facts']=audit_facts(messages,events)
        if row['reward'] != 1 and (path/'evaluation.json').exists():
            evidence=json.loads((path/'evaluation.json').read_text())
            failures.append({'run_id':row['run_id'],'task':row['task'],
                'candidate':first_unmatched(evidence['actual']['actions'],evidence['task']['actions'])})
    totals={key:{'runs':sum(bool(r[key]) for r in audits),'events':sum(len(r[key]) for r in audits)}
            for key in ('preauth_data_access','text_and_tool','multiple_tool_calls')}
    totals['authentication']={'applicable_runs':sum(not r['no_personal_data_access'] for r in audits),
        'no_data_access_runs':sum(r['no_personal_data_access'] for r in audits)}
    totals['retail_facts']=fact_totals(audits)
    errors=[(r,e) for r in audits for e in r['numeric_facts'] if e['verdict']=='violation']
    totals['numeric']={'runs':len({r['run_id'] for r,e in errors}),
                       'tasks':len({r['task'] for r,e in errors}),'events':len(errors),
                       'visible_runs':len({r['run_id'] for r,e in errors if e['user_visible']}),
                       'visible_tasks':len({r['task'] for r,e in errors if e['user_visible']}),
                       'visible_events':sum(e['user_visible'] for r,e in errors),
                       'card_fact_runs':sum(any(e['kind']=='card_last_four' for e in r['numeric_facts']) for r in audits),
                       'card_fact_events':sum(e['kind']=='card_last_four' for r in audits for e in r['numeric_facts']),
                       'visible_card_fact_runs':sum(any(e['kind']=='card_last_four' and e['user_visible'] for e in r['numeric_facts']) for r in audits),
                       'visible_card_fact_events':sum(e['kind']=='card_last_four' and e['user_visible'] for r in audits for e in r['numeric_facts']),
                       'unverified_events':sum(e['verdict']=='unverified' for r in audits for e in r['numeric_facts'])}
    dump(folder/'compliance.json',{'totals':totals,'runs':audits})
    dump(folder/'failure_candidates.json',failures)
    return totals,audits,failures


def report(folder):
    folder=Path(folder)
    rows=json.loads((folder/'runs.json').read_text())
    protocol=json.loads((folder/'protocol.json').read_text())
    metrics=summarize(rows,protocol['jobs'])
    totals,audits,failures=audit_runs(folder,rows)
    variant=protocol['config']['variant']
    if variant=='baseline':
        reference=json.loads((ROOT/'docs/results/m1_zero_20261030_145330/summary.json').read_text())
        reference_label='M1-0参考'
    else:
        previous=json.loads((ROOT/'docs/results/m1_baseline_20261031_195723/summary.json').read_text())
        reference=previous['metrics']['baseline']
        reference_label='Runtime Baseline'
    baseline=metrics[variant]
    if set(baseline['per_task_successes'])!=set(reference['per_task_successes']):
        raise ValueError('Full-pool comparison requires the same task set; use eval.metrics.summarize for targeted debugging')
    budget=json.loads((folder/'budget.json').read_text()) if (folder/'budget.json').exists() else None
    difference=baseline['pass1']-reference['pass1'] if baseline['complete'] else None
    delta=flips(reference['per_task_successes'],baseline['per_task_successes']) if baseline['complete'] else []
    harness={key:sum((r.get('harness') or {}).get(key,0) for r in rows) for key in
             ('blocked_messages','blocked_facts','rewrite_requests','extra_agent_calls','extra_steps','fail_open_messages')}
    harness.update({key:sum(r['harness_audit'][key] for r in audits) for key in
                    ('correct_interceptions','false_interceptions','unknown_interceptions','extra_http_attempts')})
    dump(folder/'summary.json',{'metrics':metrics,'reference_pass1':reference['pass1'],
        'evaluation_implementation_commit':git(ROOT,'rev-parse','HEAD').strip(),
        'reference_pass3':reference['pass3'],'pass1_difference':difference,
        'task_flips_vs_reference':delta,'compliance':totals,'budget':budget,
        'execution_revisions':protocol.get('execution_revisions',[]),'harness':harness,
        'extractor_shared_with_harness':True,'F1_independent_card_audit_sample_size':10})
    ref_cost=f"{reference['estimated_cost_cny']:.4f}" if reference['estimated_cost_cny'] is not None else '未知'
    lines=[f'# 开发池结果：{variant}','','| 配置 | 完成 | pass¹ | pass³ | 折算费用（元） |',
           '|---|---:|---:|---:|---:|',
           f"| {reference_label} |120|{reference['pass1']:.2%}|{reference['pass3']:.2%}|{ref_cost}|"]
    if baseline['complete']:
        cost=f"{baseline['estimated_cost_cny']:.4f}" if baseline['estimated_cost_cny'] is not None else '未知'
        lines.append(f"| Runtime |{baseline['completed']}|{baseline['pass1']:.2%}|{baseline['pass3']:.2%}|{cost}|")
    else: lines.append('批次未完成，正式pass指标留空。')
    lines+=['','## 自动检查（开发池）','',json.dumps(totals,ensure_ascii=False,indent=2),
            '','## 相对参考的任务成功次数变化','']
    lines += [f"- {r['task']}: {r['before']}/3 → {r['after']}/3" for r in delta]
    if budget:
        lines+=['','## 费用与调用配置','',
            f"已知折算费用小计：{budget['known_equivalent_cny']:.6f}元；费用未知请求：{budget['unknown_request_count']}次。",
            f"调度预留：{budget['scheduling_reserve_cny']:.2f}元，独立于实测费用。",
            '调用配置见protocol.json；H组与冻结后的Baseline将共用同一套传输和重试配置。']
    if variant!='baseline':
        lines+=['','## H拦截与额外开销','',json.dumps(harness,ensure_ascii=False,indent=2),
            '','主指标只统计实际发给用户的错误；被拦文本保留为原始输出。付款标识误拦对照独立初始支付元数据，H2误拦逐条对照原始对话与工具关系（fact_block_review.json），不以H反馈自己的expected字段判分。',
            '额外Agent调用为H触发至下一次实际发送之间的决策调用，包含补查工具期间的调用；额外步数为这些调用产生的成功决策轮。',
            'H与评测共用纯文本提取器；F1完成后须从H开启组抽10条卡号轨迹，不调用提取器逐条核对，随F1汇报。']
    (folder/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return metrics,totals
