from copy import deepcopy
import json
from domains.tau.payment_hook import payment_feedback
from eval.payment_audit import audit_interceptions
from runtime.agent import run
from runtime.types import AgentState, Usage, ModelReply, ToolResult, ToolStatus

BAD='Your Visa card ending 6173085 will receive the refund.'
GOOD='Your Visa card ending 4491 will receive the refund.'
PROFILE={'role':'tool','name':'get_user_details','tool_call_id':'profile',
         'content':json.dumps({'payment_methods':{'credit_card_6173085':{'last_four':'4491'}}})}


class Model:
    def __init__(self, replies): self.replies,self.calls=iter(replies),[]
    def complete(self,messages,tools,*,timeout_s):
        self.calls.append(deepcopy(messages))
        value=next(self.replies)
        message=value if isinstance(value,dict) else {'role':'assistant','content':value}
        return ModelReply(message,Usage(),deepcopy(message))


class Env:
    mode='dialogue';done=False
    def __init__(self):self.sent=[]
    def list_tools(self):return []
    def execute(self,call):return ToolResult(ToolStatus.OK,PROFILE['content'])
    def respond(self,text):self.sent.append(text);self.done=True;return '###STOP###'


def execute(replies,enabled):
    state=AgentState('h',messages=[{'role':'system','content':'policy'},
        {'role':'assistant','content':None,'tool_calls':[{'id':'profile','function':{'name':'get_user_details','arguments':'{}'}}]},PROFILE])
    env,model,events=Env(),Model(replies),[]
    result=run(model,env,env,system='policy',state=state,
        before_respond=payment_feedback if enabled else None,
        emit=lambda event,**data:events.append({'event':event,**data}))
    return result,env,model,events


def test_flag_off_is_original_baseline_behavior():
    result,env,model,events=execute([BAD],False)
    assert env.sent==[BAD] and len(model.calls)==1 and result.state.hook_state=={}


def test_error_is_withheld_and_model_rewrites_from_observed_facts():
    result,env,model,events=execute([BAD,GOOD],True)
    assert env.sent==[GOOD] and model.calls[1][-2]['content']==BAD
    assert '4491' in model.calls[1][-1]['content']
    assert result.state.step==2
    assert result.state.hook_state['payment']['extra_agent_calls']==1
    assert result.state.hook_state['payment']['extra_steps']==1
    assert [e['event'] for e in events]==['h_block','h_rewrite_accepted','user_turn']


def test_correct_card_gift_id_and_unknown_last_four_do_not_invent_facts():
    assert payment_feedback([PROFILE],GOOD+' Gift card ending 5701566.') is None
    order={'role':'tool','name':'get_order_details','content':json.dumps({'payment_history':[{'payment_method_id':'credit_card_6173085'}]})}
    feedback=payment_feedback([order],BAD)
    assert feedback['problems'][0]['correct_last_four'] is None
    assert feedback['problems'][0]['correct_known_value']=='credit_card_6173085'
    assert '4491' not in feedback['text']


def test_after_two_rewrites_original_third_text_is_sent_and_traced():
    result,env,model,events=execute([BAD,BAD,BAD],True)
    assert env.sent==[BAD] and len(model.calls)==3
    assert result.state.hook_state['payment']['blocked_messages']==2
    assert result.state.hook_state['payment']['fail_open_messages']==1
    assert [e['text'] for e in events if e['event']=='h_fail_open']==[BAD]


def test_evaluator_excludes_withheld_text_and_uses_own_truth_for_false_blocks():
    facts=[{'message_index':4,'user_visible':True},{'message_index':6,'user_visible':True}]
    trace=[{'event':'h_block','message_index':4,'text':'card ending 4491',
        'problems':[{'payment_method_id':'credit_card_6173085','claimed_value':'4491','correct_last_four':'WRONG H FEEDBACK'}]}]
    trace += [{'event':'model_call','actor':'agent'}, {'event':'model_call','actor':'agent'},
              {'event':'h_rewrite_accepted'}, {'event':'model_call','actor':'agent'}]
    audit=audit_interceptions(facts,trace,{'credit_card_6173085':{'last_four':'4491'}})
    assert not facts[0]['user_visible'] and facts[1]['user_visible']
    assert audit['false_interceptions']==1 and audit['correct_interceptions']==0
    assert audit['extra_http_attempts']==2


def test_missing_card_digits_can_be_queried_between_block_and_rewrite():
    order={'role':'tool','name':'get_order_details','content':json.dumps({'user_id':'u',
        'payment_history':[{'payment_method_id':'credit_card_6173085'}]})}
    state=AgentState('query',messages=[{'role':'system','content':'policy'},order])
    query={'role':'assistant','content':None,'tool_calls':[{'id':'q','type':'function',
        'function':{'name':'get_user_details','arguments':'{"user_id":"u"}'}}]}
    model,env=Model([BAD,query,GOOD]),Env()
    result=run(model,env,env,system='policy',state=state,before_respond=payment_feedback)
    assert env.sent==[GOOD] and result.state.step==3
    assert result.state.hook_state['payment']['blocked_messages']==1
    assert result.state.hook_state['payment']['extra_agent_calls']==2
