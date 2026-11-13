from scripts.m1_zero import plan_runs, scores, saturation
from m0.protocol import build_manifest
from pathlib import Path
import pytest
import json

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / 'configs/m1_zero.json').read_text())


def test_three_sequential_rounds_use_only_the_forty_development_tasks():
    manifest = build_manifest(ROOT / 'vendor/tau-bench')
    jobs = plan_runs(manifest)
    assert len(jobs) == 120
    for round_number in (1, 2, 3):
        actual = [r['task_ref'] for r in jobs if r['round'] == round_number]
        assert actual == manifest['development']
        assert {t['split'] for t in actual} == {'dev', 'train'}
    assert len({r['run_id'] for r in jobs}) == 120


def test_scores_use_success_counts_and_keep_partial_batches_incomplete():
    rows = [{'task': task, 'round': r, 'reward': reward}
            for task, rewards in [('a', [1, 1, 0]), ('b', [1, 1, 1])]
            for r, reward in enumerate(rewards, 1)]
    result = scores(rows, expected=6)
    assert result['complete']
    assert result['pass1'] == pytest.approx(5 / 6)
    assert result['pass3'] == 0.5
    assert result['pass1_minus_pass3'] == pytest.approx(5 / 6 - 0.5)
    assert scores(rows[:-1], expected=6)['pass1'] is None


def test_saturation_uses_distinct_tasks_excludes_category_ten_and_requires_audit():
    repair = [{'task': f't{i}', 'category': 1} for i in range(4)]
    assert saturation(repair, [], POLICY)['rule'] == 'continue_retail_success_rate'
    duplicates = [{'task': 't0', 'category': 1}] * 4
    tenth = [{'task': f'other{i}', 'category': 10} for i in range(8)]
    assert saturation(duplicates + tenth, [], POLICY)['rule'] == 'pending_success_audit'
    audits = [{'policy_violation': i < 3} for i in range(10)]
    assert saturation(duplicates + tenth, audits, POLICY)['rule'] == 'compliance_direction'
    assert saturation([], [{'policy_violation': False} for _ in range(10)], POLICY)['rule'] == 'consider_airline'
    assert saturation([{'task': 't0', 'category': None}], audits, POLICY)['rule'] == 'pending_failure_labels'
