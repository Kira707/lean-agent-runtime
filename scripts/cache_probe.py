"""Replay existing requests once with one shared key; do not execute tools."""

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path

from m0.protocol import load_config
from m0.recording import CallRecorder

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("docs/results/pilot_20261029_193917/calls.jsonl")
TASK = "retail/dev/4"


def replay_requests(rows, recorder, complete):
    for index, old in enumerate(rows, 1):
        recorder.task, recorder.phase = old["task"], old["phase"]
        recorder.call(complete, old["actor"], **copy.deepcopy(old["request"]))
        print(f"Replayed {index}/{len(rows)}: {old['actor']}", flush=True)


def statistics(rows):
    prompt = sum(r["input_tokens"] for r in rows)
    cached = sum(r["cached_input_tokens"] for r in rows)
    output = sum(r["output_tokens"] for r in rows)
    input_cost = (cached * 0.02 + (prompt - cached)) / 1_000_000
    output_cost = output * 4 / 1_000_000
    return {"calls": len(rows), "input_tokens": prompt, "cached_tokens": cached,
            "cache_ratio": cached / prompt, "output_tokens": output,
            "input_cost_cny": input_cost, "output_cost_cny": output_cost,
            "total_cost_cny": input_cost + output_cost}


def save_comparison(folder, original, replayed):
    comparison = {mode: {actor: statistics([r for r in rows if actor == "all" or r["actor"] == actor])
                        for actor in ("all", "agent", "user")}
                  for mode, rows in (("original_two_keys", original), ("shared_key_replay", replayed))}
    comparison["identical_recorded_requests"] = all(
        a["request"] == b["request"] for a, b in zip(original, replayed))
    (folder / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    lines = ["# 双Key历史记录与共享Key轨迹重放", "", f"任务：{TASK}；历史来源：{SOURCE}。",
        "按原顺序重放每个已保存请求，不执行工具；新输出不参与后续请求。",
        "只有共享Key组是本次新运行，历史组的预热状态、请求时点与节奏不同；不是严格因果对照。",
        "价格为指定空闲价格折算，不代表Coding Plan实际扣费。", "",
        "| 组别 | 调用方 | 输入token | 缓存token | 命中占比 | 输出token | 输入费用/元 | 输出费用/元 | 合计/元 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in ("original_two_keys", "shared_key_replay"):
        for actor, s in comparison[mode].items():
            lines.append(f"| {mode} | {actor} | {s['input_tokens']} | {s['cached_tokens']} | {s['cache_ratio']:.2%} | "
                         f"{s['output_tokens']} | {s['input_cost_cny']:.8f} | {s['output_cost_cny']:.8f} | {s['total_cost_cny']:.8f} |")
    lines += ["", f"两组记录的请求内容一致：{comparison['identical_recorded_requests']}。", "",
              "## 新Key首次请求", ""]
    for actor in ("user", "agent"):
        first = next(r for r in replayed if r["actor"] == actor)
        lines.append(f"- {actor}：输入{first['input_tokens']}，缓存{first['cached_input_tokens']}。")
    (folder / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison


def run(root=ROOT, *, live=False):
    if not live:
        raise ValueError("Shared-key paid replay requires explicit --live")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    from dotenv import load_dotenv
    load_dotenv(root / ".env")
    if not os.environ.get("ARK_CACHE_TEST_API_KEY"):
        raise ValueError("Missing ARK_CACHE_TEST_API_KEY")
    import litellm
    config = load_config(root / "configs/m0_pilot.json")
    for actor in ("agent", "user"):
        config[actor]["api_key_env"] = "ARK_CACHE_TEST_API_KEY"
    config["cost_cap_cny"] = "1"
    original = [json.loads(s) for s in (root / SOURCE).read_text(encoding="utf-8").splitlines()
                if json.loads(s)["task"] == TASK]
    folder = root / "docs/results" / datetime.now().strftime("cache_probe_%Y%m%d_%H%M%S")
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    recorder = CallRecorder(folder / "calls.jsonl", config)
    replay_requests(original, recorder, litellm.completion)
    result = save_comparison(folder, original, recorder.records)
    print(json.dumps(result, indent=2))
    print(f"Comparison artifacts: {folder}")
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    run(live=parser.parse_args().live)
