import copy
import json
from pathlib import Path

import pytest

from m0.protocol import load_config
from m0.recording import CallRecorder
from scripts.cache_probe import ROOT, SOURCE, TASK, replay_requests, run


def test_replay_preserves_original_requests_and_does_not_feed_new_output(tmp_path, monkeypatch):
    original = [json.loads(s) for s in (ROOT / SOURCE).read_text(encoding='utf-8').splitlines()
                if json.loads(s)['task'] == TASK]
    unchanged = copy.deepcopy(original)
    config = load_config(ROOT / 'configs/m0_pilot.json')
    for actor in ('agent', 'user'):
        config[actor]['api_key_env'] = 'ARK_CACHE_TEST_API_KEY'
    monkeypatch.setenv('ARK_CACHE_TEST_API_KEY', 'test-only-key')
    recorder = CallRecorder(tmp_path / 'calls.jsonl', config)
    def complete(**kwargs):
        class Response:
            _hidden_params = {}
            def model_dump(self, **kwargs):
                return {'choices': [{'message': {'content': 'DIFFERENT NEW OUTPUT'}}],
                        'usage': {'prompt_tokens': 100, 'completion_tokens': 10,
                                  'prompt_tokens_details': {'cached_tokens': 0}}}
        return Response()
    replay_requests(original, recorder, complete)
    assert len(recorder.records) == 12
    assert original == unchanged
    assert [r['request'] for r in recorder.records] == [r['request'] for r in original]
    assert [r['actor'] for r in recorder.records] == [r['actor'] for r in original]
    assert 'test-only-key' not in recorder.path.read_text()


def test_paid_probe_requires_live_flag():
    with pytest.raises(ValueError, match='--live'):
        run(live=False)
