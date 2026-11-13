"""Project-side call records; prices are user-specified equivalents, not invoices."""

import json
import os
import time
from decimal import Decimal
from pathlib import Path


def cost_from_usage(usage: dict, prices: dict) -> dict:
    prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
    result = {"input_tokens": prompt, "output_tokens": output, "cached_input_tokens": None,
              "uncached_input_tokens": None, "estimated_cost_cny": None,
              "cost_upper_bound_cny": None, "cost_unknown_reason": None}
    if type(prompt) is not int or type(output) is not int or min(prompt, output) < 0:
        result["cost_unknown_reason"] = "Missing or invalid prompt/completion token usage"
        return result
    hit = usage.get("prompt_cache_hit_tokens")
    if hit is None:
        hit = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if miss is not None and type(miss) is not int:
        result["cost_unknown_reason"] = "Invalid cache miss token usage"
        return result
    if hit is None and miss is not None:
        hit = prompt - miss
    unit = Decimal(prices["unit_tokens"])
    out_cost = Decimal(output) * Decimal(prices["output"])
    result["cost_upper_bound_cny"] = float((Decimal(prompt) * Decimal(prices["uncached_input"]) + out_cost) / unit)
    if type(hit) is not int or not 0 <= hit <= prompt or (miss is not None and miss != prompt - hit):
        result["cost_unknown_reason"] = "Cache token usage missing or inconsistent; upper bound uses all uncached input"
        return result
    miss = prompt - hit
    total = (Decimal(hit) * Decimal(prices["cached_input"]) +
             Decimal(miss) * Decimal(prices["uncached_input"]) + out_cost) / unit
    result.update(cached_input_tokens=hit, uncached_input_tokens=miss, estimated_cost_cny=float(total))
    return result


class CallRecorder:
    """Mutable pilot data only; not a Runtime interface or plugin abstraction."""

    def __init__(self, path: Path, config: dict):
        self.path, self.config = path, config
        self.task, self.phase = None, "task"
        self.records = []
        self.spent_upper = Decimal(0)
        self.pending = None

    def call(self, complete, actor: str, **kwargs):
        if self.spent_upper >= Decimal(self.config["cost_cap_cny"]):
            raise RuntimeError("pilot_budget_reached")
        route = self.config[actor]
        request = kwargs | {"model": route["model"], "custom_llm_provider": route["provider"],
            "api_base": route["api_base"], "api_key": os.environ.get(route["api_key_env"]),
            "temperature": route["temperature"], "timeout": self.config["request_timeout_seconds"],
            "num_retries": 0, "max_retries": 0, "max_tokens": self.config["max_output_tokens"]}
        started = time.monotonic()
        self.pending = {"task": self.task, "actor": actor, "phase": self.phase,
                        "request": {k: v for k, v in request.items() if k != "api_key"}}
        response = complete(**request)
        raw = response.model_dump(mode="json")
        cost = cost_from_usage(raw.get("usage") or {}, self.config["prices"])
        hidden = getattr(response, "_hidden_params", {})
        record = {"task": self.task, "actor": actor, "phase": self.phase,
                  "wall_ms": round((time.monotonic() - started) * 1000),
                  "request": {k: v for k, v in request.items() if k != "api_key"},
                  "response": raw, "litellm_cost_reference": hidden.get("response_cost"),
                  "cost_basis": self.config["prices"]["basis"], **cost}
        self.records.append(record)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.pending = None
        upper = cost["cost_upper_bound_cny"]
        if upper is None:
            raise RuntimeError("pilot_usage_missing; stop before further paid requests")
        self.spent_upper += Decimal(str(upper))
        return response

    def record_failure(self, error):
        if self.pending is not None:
            record = self.pending | {"response": None, "input_tokens": None,
                "output_tokens": None, "estimated_cost_cny": None,
                "cost_unknown_reason": "Request failed; billed usage is unknown",
                "error_type": type(error).__name__, "error": str(error)}
            self.records.append(record)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.pending = None
