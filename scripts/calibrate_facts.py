"""Predefined mutations of real development statements; zero model calls."""
from copy import deepcopy
import json
from pathlib import Path
from domains.tau.fact_hook import fact_feedback
from eval.runner import dump
from runtime.agent import run
from runtime.tracer import Tracer
from runtime.types import AgentState, ModelReply, Usage

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/results/m1_baseline_20261031_195723'


def saved(run_id, index):
    messages=json.loads((BASE/run_id/'messages.json').read_text())
    return messages[:index-1],messages[index-1]['content']


def calibrate(folder):
    folder=Path(folder)
    history,text=saved('baseline_r01_train_138',17)
    # Expected violations are specified before calling H2; no generated labels.
    cases=[('SKU price',history,text.replace('$150.58','$137.32'),'item_price'),
        ('SKU connectivity',history,text.replace('black / laser / wireless','black / laser / wired'),'item_attribute:connectivity'),
        ('price difference',history,text.replace('$13.26','$137.32'),'amount:difference'),
        ('refund direction',history,text.replace('paid by you via PayPal','refunded to you via PayPal'),'amount:direction'),
        ('wrong replacement SKU',history,text.replace('item 8214883393','item 3330317167'),'item_price')]
    h,t=saved('baseline_r01_dev_013',15)
    cases.append(('refund total',h,t.replace('$766.54','$194.47'),'amount:refund'))
    results=[]
    for label,h,t,expected_kind in cases:
        feedback=fact_feedback(h,t)
        kinds=[p['kind'] for p in feedback['problems']] if feedback else []
        results.append({'case':label,'expected_error_kind':expected_kind,'detected_kinds':kinds,
                        'pass':expected_kind in kinds,'text':t})
    boundaries=[]
    for run_id,index in [('baseline_r01_dev_009',21),('baseline_r03_dev_001',23),
                         ('baseline_r01_dev_019',21),('baseline_r02_train_092',33),
                         ('baseline_r03_dev_004',13),('baseline_r01_dev_012',49)]:
        h,t=saved(run_id,index)
        boundaries.append({'run_id':run_id,'message_index':index,'text':t,
                           'pass':fact_feedback(h,t) is None})
    dump(folder/'fact_calibration.json',{'method':'Six predefined error mutations and six correct boundary statements checked by the assistant against saved tool data; no live API calls. Not a human-labelled natural-error recall estimate.',
        'mutations':results,'boundaries':boundaries})
    path=folder/'offline_interception_demo';path.mkdir(exist_ok=True)
    trace=Tracer(path/'trace.jsonl','offline_h2_demo')
    if trace.path.exists():trace.path.unlink()
    trace('run_start',mode='offline_scripted_mutation',source_run='baseline_r01_train_138',source_message_index=17)
    class Model:
        def __init__(self):self.texts=iter([text.replace('$13.26','$137.32'),text])
        def complete(self,*args,**kwargs):
            return ModelReply({'role':'assistant','content':next(self.texts)},Usage(),{})
    class Env:
        mode='dialogue';done=False
        def list_tools(self):return []
        def respond(self,value):self.done=True;return '###STOP###'
    env=Env()
    result=run(Model(),env,env,system='policy',state=AgentState('offline_h2_demo',messages=deepcopy(history)),
               before_respond=fact_feedback,emit=trace)
    dump(path/'messages.json',result.state.messages)
    trace('run_end',mode='offline_scripted_mutation',reward=None,harness=result.state.hook_state)
    return results,boundaries


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('folder');args=parser.parse_args()
    results,boundaries=calibrate(args.folder)
    print(f'Injected errors detected: {sum(r["pass"] for r in results)}/{len(results)}; '
          f'correct boundaries accepted: {sum(r["pass"] for r in boundaries)}/{len(boundaries)}')
