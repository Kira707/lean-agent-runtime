import copy
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from m0.protocol import build_manifest, load_config, validate_manifest, verify_base
from m0.recording import CallRecorder, cost_from_usage

ROOT = Path(__file__).resolve().parents[1]
TAU = ROOT / "vendor/tau-bench"


class Response:
    def __init__(self, content="hello", cost=0.25):
        self.choices = [SimpleNamespace(message=SimpleNamespace(
            content=content, model_dump=lambda: {"role": "assistant", "content": content}))]
        self._hidden_params = {"response_cost": cost}

    def model_dump(self, **kwargs):
        return {"id": "saved-response", "choices": [{"message": {
            "role": "assistant", "content": self.choices[0].message.content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                      "prompt_tokens_details": {"cached_tokens": 40}}}


def test_manifest_is_deterministic():
    manifest = build_manifest(TAU)
    assert manifest == build_manifest(TAU)
    assert len(manifest["development"]) == 40
    assert len(manifest["test"]) == 50
    assert len(manifest["pilot"]) == 5
    assert len(manifest["airline_holdout"]) == 50


def test_task_reference_boundaries():
    manifest = build_manifest(TAU)
    for change in ("duplicate", "out_of_range"):
        bad = copy.deepcopy(manifest)
        if change == "duplicate":
            bad["development"].append(bad["development"][0])
        else:
            bad["development"][0]["index"] = 99999
        with pytest.raises(ValueError):
            validate_manifest(bad)
    same_index = copy.deepcopy(manifest)
    same_index["development"] = [{"domain": "retail", "split": split, "index": 0}
                                 for split in ("dev", "train")]
    same_index["pilot"] = [same_index["development"][0]]
    validate_manifest(same_index)


def test_pilot_rejects_test_and_nonmember():
    for split in ("test", "train"):
        bad = build_manifest(TAU)
        bad["pilot"][0] = {"domain": "retail", "split": split, "index": 499 if split == "train" else 0}
        with pytest.raises(ValueError):
            validate_manifest(bad)


def test_temperature_is_forwarded_and_user_config_frozen(monkeypatch, tmp_path):
    import tau_bench.envs.user as user_module
    from tau_bench.envs import get_env

    seen = []
    def complete(**kwargs):
        seen.append(kwargs)
        return Response()
    monkeypatch.setattr(user_module, "completion", complete)
    get_env("retail", "llm", "test-model", "dev", "openai", 0, user_temperature=0.15)
    assert seen[0]["temperature"] == 0.15
    config = json.loads((ROOT / "configs/m0_pilot.json").read_text())
    config["user"]["model"] = "changed"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="frozen"):
        load_config(path, ROOT / "configs/user_simulator.lock.json")


def test_simulator_cost_accumulates(monkeypatch):
    import tau_bench.envs.user as user_module

    costs = iter([0.1, 0.2, 0.3])
    monkeypatch.setattr(user_module, "completion", lambda **kw: Response(cost=next(costs)))
    user = user_module.LLMUserSimulationEnv("model", "openai", temperature=0)
    user.reset("task")
    user.step("next")
    assert user.get_total_cost() == pytest.approx(0.6)


def test_reset_keeps_cost_and_instances_are_isolated(monkeypatch):
    import tau_bench.envs.user as user_module

    monkeypatch.setattr(user_module, "completion", lambda **kw: Response())
    first = user_module.LLMUserSimulationEnv("model", "openai")
    second = user_module.LLMUserSimulationEnv("model", "openai")
    first.reset("task")
    assert first.get_total_cost() == 0.5
    assert second.get_total_cost() == 0.25
    assert len(second.messages) == 3


def test_initialization_is_recorded(tmp_path):
    recorder = CallRecorder(tmp_path / "calls.jsonl", load_config(ROOT / "configs/m0_pilot.json"))
    recorder.task = "retail/dev/0"
    recorder.phase = "initialization"
    recorder.call(lambda **kwargs: Response(), "user", messages=[])
    record = json.loads(recorder.path.read_text())
    assert record["phase"] == "initialization"
    assert record["task"] == "retail/dev/0"
    assert record["estimated_cost_cny"] is not None


def test_recorder_preserves_response_and_separates_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_USER_API_KEY", "user-secret")
    monkeypatch.setenv("ARK_AGENT_API_KEY", "agent-secret")
    config = load_config(ROOT / "configs/m0_pilot.json")
    recorder = CallRecorder(tmp_path / "calls.jsonl", config)
    response, seen = Response(), []
    def complete(**kwargs):
        seen.append(kwargs)
        return response
    assert recorder.call(complete, "user", messages=[{"role": "user", "content": "hi"}]) is response
    recorder.call(complete, "agent", messages=[])
    assert [x["api_key"] for x in seen] == ["user-secret", "agent-secret"]
    assert seen[0]["api_base"] == config["user"]["api_base"]
    text = recorder.path.read_text()
    records = [json.loads(line) for line in text.splitlines()]
    assert records[0]["response"] == response.model_dump()
    assert records[0]["request"]["messages"][0]["content"] == "hi"
    assert "user-secret" not in text and "agent-secret" not in text


def test_unknown_cost_is_not_zero(tmp_path):
    prices = load_config(ROOT / "configs/m0_pilot.json")["prices"]
    missing = cost_from_usage({}, prices)
    assert missing["estimated_cost_cny"] is None
    assert missing["cost_unknown_reason"]
    no_cache = cost_from_usage({"prompt_tokens": 100, "completion_tokens": 20}, prices)
    assert no_cache["estimated_cost_cny"] is None
    assert no_cache["cost_upper_bound_cny"] > 0


def test_offline_test_network_is_blocked():
    with pytest.raises(RuntimeError, match="Network forbidden"):
        socket.create_connection(("example.com", 443))


def test_patch_does_not_touch_scoring_tasks_policies_or_tools():
    fingerprint = verify_base(TAU, ROOT / "patches/tau_user_simulator.patch")
    assert fingerprint["commit"] == "59a200c6d575d595120f1cb70fea53cef0632f6b"
    patch = (ROOT / "patches/tau_user_simulator.patch").read_text()
    assert "def calculate_reward" not in patch and "def step(self, action" not in patch
    assert not any(x in patch for x in ["tasks_test.py", "wiki.md", "/tools/"])


def test_cache_price_accounting():
    prices = load_config(ROOT / "configs/m0_pilot.json")["prices"]
    base = {"prompt_tokens": 1000000, "completion_tokens": 1000000}
    for cache_fields in ({"prompt_cache_hit_tokens": 250000},
                         {"prompt_tokens_details": {"cached_tokens": 250000}}):
        result = cost_from_usage(base | cache_fields, prices)
        assert result["estimated_cost_cny"] == pytest.approx(4.755)
        assert result["cached_input_tokens"] == 250000
        assert result["uncached_input_tokens"] == 750000


def test_pilot_default_is_offline_and_preserves_raw_response(tmp_path):
    from scripts.m0 import run_cost_pilot

    with pytest.raises(ValueError, match="--live"):
        run_cost_pilot(ROOT, live=False)
    recorder = CallRecorder(tmp_path / "calls.jsonl", load_config(ROOT / "configs/m0_pilot.json"))
    recorder.call(lambda **kw: Response("full content"), "agent", messages=[])
    saved = json.loads(recorder.path.read_text())
    assert saved["response"]["choices"][0]["message"]["content"] == "full content"
