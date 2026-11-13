from runtime.tracer import Tracer
from runtime.store import RunStore
from runtime.types import ToolCall, ToolResult, ToolStatus


def test_event_order_and_serializable_action(tmp_path):
    trace=Tracer(tmp_path/'trace.jsonl','r1')
    trace('run_start',config={'name':'baseline'})
    trace('model_call',actor='user',phase='initialization',usage={'input_tokens':10})
    trace('tool_call',call=ToolCall('a','lookup',{}),result=ToolResult(ToolStatus.OK,'ok'))
    trace('user_turn',text='done',observation='STOP')
    trace('run_end',status='user_ended',reward=None)
    rows=trace.read()
    assert [r['event'] for r in rows]==['run_start','model_call','tool_call','user_turn','run_end']
    assert all(r['run_id']=='r1' for r in rows) and rows[2]['call']['name']=='lookup'


def test_runs_and_failed_attempts_are_separate(tmp_path):
    first,second=Tracer(tmp_path/'a.jsonl','a'),Tracer(tmp_path/'b.jsonl','b')
    first('model_call',actor='user',phase='initialization',attempt=0,usage=None,error='timeout')
    first('model_call',actor='user',phase='initialization',attempt=1,usage={'input_tokens':1})
    second('model_call',actor='agent',usage={'input_tokens':2})
    assert len(first.read())==2 and len(second.read())==1
    assert first.read()[0]['usage'] is None


def test_store_reopen_keeps_missing_reward_and_trace(tmp_path):
    path=tmp_path/'runs.sqlite'
    row={'run_id':'r1','status':'budget_exceeded','reward':None,'trace_path':'r1/trace.jsonl'}
    RunStore(path).save(row)
    reopened=RunStore(path)
    assert reopened.get('r1')==row and reopened.all()==[row] and reopened.get('missing') is None
