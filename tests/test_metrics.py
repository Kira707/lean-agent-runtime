import pytest
from eval.metrics import summarize, flips
from eval.divergence import first_unmatched


def rows():
    return [{'run_id':str(i),'variant':'baseline','task':'retail/dev/4','trial':i,
             'status':'user_ended','reward':int(i<2),'usage':{'estimated_cost_cny':.1}} for i in range(3)]


def test_known_two_of_three_pass_values():
    data=rows(); result=summarize(data,data)['baseline']
    assert result['pass1']==pytest.approx(2/3) and result['pass2']==pytest.approx(1/3) and result['pass3']==0


def test_flips_pair_by_task_success_count():
    assert flips({'a':0,'b':3},{'a':2,'b':3})==[{'task':'a','before':0,'after':2}]


def test_missing_trial_not_started_and_unknown_reward():
    data=rows(); assert summarize(data[:2],data)['baseline']['pass1'] is None
    data[2]['status']='not_started'; assert summarize(data,data)['baseline']['pass3'] is None
    data[2].update(status='model_error',reward=None)
    assert summarize(data,data)['baseline']['successes']==2 and data[2]['reward'] is None
    data[2]['usage']['estimated_cost_cny']=None
    assert summarize(data,data)['baseline']['estimated_cost_cny'] is None
    data[2]['wall_ms']=None
    assert summarize(data,data)['baseline']['wall_sample_count']==0


def test_divergence_candidate_not_error_and_reads_can_reorder():
    gold=[{'name':'write','kwargs':{'id':1}}]
    actual=[{'name':'read_b'},{'name':'read_a'},*gold]
    assert first_unmatched(actual,gold) is None
    assert first_unmatched([],gold)['classification'] is None
