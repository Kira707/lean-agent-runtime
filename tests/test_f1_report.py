import json
from pathlib import Path
from eval.payment_audit import audit_interceptions
from eval.runner import dump


def test_combined_false_count_examples_unknowns_and_independent_sample(monkeypatch,tmp_path):
    import eval.f1_report as module
    source=Path(__file__).resolve().parents[1]/'docs/results/m1_fact_guard_20261101_195618/fact_guard_r03_train_400/trace.jsonl'
    block=next(e for e in map(json.loads,source.read_text().splitlines()) if e['event']=='h_block')
    reason='154.75 + 54.04 is the total refund 208.79, not a single item price.'
    rows=[];audits=[]
    for variant in ('baseline','payment_guard','all_guards'):
        for trial in (1,2,3):
            run_id=f'{variant}_{trial}'
            row={'run_id':run_id,'variant':variant,'trial':trial,'task':'retail/test/0',
                'status':'user_ended','reward':1,'steps':1,'wall_ms':1,
                'usage':{'estimated_cost_cny':0},'trace_path':str(tmp_path/run_id/'trace.jsonl')}
            rows.append(row)
            trace=[block] if variant=='all_guards' and trial in (1,2) else []
            reviews={str(block['message_index']):{'verdict':'false','confidence':'high','reason':reason}} if trial==1 else None
            audit=audit_interceptions([],trace,{},reviews)
            audits.append({'run_id':run_id,'numeric_facts':[],'retail_facts':[], 'harness_audit':audit})
    dump(tmp_path/'runs.json',rows)
    dump(tmp_path/'protocol.json',{'jobs':rows,'core':{'audit':{'seed':20261031}}})
    monkeypatch.setattr(module,'audit_runs',lambda folder,records:({},audits,[]))
    _,compliance=module.report_f1(tmp_path)
    combined=compliance['all_guards']
    assert combined['false_interceptions']==1 and combined['unknown_interceptions']==1
    example=combined['false_interception_examples'][0]
    assert example['mechanisms']==['H2'] and example['review']['reason']==reason
    assert example['text']==block['text'] and example['message_index']==block['message_index']
    text=(tmp_path/'summary.md').read_text()
    assert 'H1+H2误拦' in text and '已确认误拦：1次；待判：1次' in text
    assert '> '+block['text'].replace('\n','\n> ') in text and reason in text
    saved=json.loads((tmp_path/'summary.json').read_text())
    assert saved['compliance']['all_guards']['false_interception_examples']==[example]
    sample=json.loads((tmp_path/'independent_audit_order.json').read_text())
    assert sample['variant']=='all_guards' and sample['required']==10
    assert sample['extractor_used_for_selection'] is False
    assert set(sample['candidate_order'])=={'all_guards_1','all_guards_2','all_guards_3'}
