"""M0 preparation and explicitly triggered reference-agent cost pilot."""

import argparse
import importlib.metadata
import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from m0.protocol import (build_manifest, git, load_config, task_key, validate_manifest,
                         verify_base, write_manifest)
from m0.recording import CallRecorder

ROOT = Path(__file__).resolve().parents[1]


def prepare(root: Path):
    tau = root / "vendor/tau-bench"
    patch_file = root / "patches/tau_user_simulator.patch"
    fingerprint = verify_base(tau, patch_file)
    if fingerprint["worktree"] == "clean":
        git(tau, "apply", "--check", str(patch_file))
        git(tau, "apply", str(patch_file))
    write_manifest(build_manifest(tau), root / "configs/task_manifest.json")
    print("M0 manifest prepared; approved simulator patch applied. No model calls.")


def write_budget(output: Path, recorder: CallRecorder, runs: list):
    known = [r["estimated_cost_cny"] for r in recorder.records]
    exact = sum(known) if all(x is not None for x in known) else None
    lines = ["# M0五题费用试点", "", "按用户给定的DeepSeek空闲价格折算，非Coding Plan实际账单。",
             "", f"模型：{recorder.config['user']['model']}；并发1；每题1次。",
             f"总折算费用：{exact if exact is not None else '未知（缓存分项或usage缺失）'} 元。",
             f"全部输入按未命中计算的费用上界：{float(recorder.spent_upper):.8f} 元。", "",
             "| 任务 | 状态 | Agent输入/输出token | 模拟器输入/输出token | 折算费用 |",
             "|---|---|---|---|---|"]
    for run in runs:
        records = [r for r in recorder.records if r["task"] == run["task"]]
        totals = {}
        for actor in ("agent", "user"):
            selected = [r for r in records if r["actor"] == actor]
            totals[actor] = "/".join(str(sum(r[k] for r in selected))
                if all(r.get(k) is not None for r in selected) else "未知"
                for k in ("input_tokens", "output_tokens"))
        costs = [r["estimated_cost_cny"] for r in records]
        amount = f"{sum(costs):.8f}" if costs and all(c is not None for c in costs) else "未知"
        lines.append(f"| {run['task']} | {run['status']} | {totals['agent']} | {totals['user']} | {amount} |")
    lines += ["", "初始化调用计入模拟器费用；完整调用记录见同目录calls.jsonl。",
              "M0关闭自动重试；embedding未发生。失败请求的未知计费不记为0。",
              "", "## M1预算外推", ""]
    if runs and all(r["status"] == "finished" for r in runs) and exact is not None:
        average = exact / len(runs)
        lines += [f"单题平均折算费用：{average:.8f} 元。",
                  f"50题×3次×Baseline及一个候选：约 {average * 300:.4f} 元（仅按试点均值外推，不是上限）。",
                  f"40题开发池×1次：约 {average * 40:.4f} 元；实际调试轮次另计。"]
    else:
        lines += ["试点未完整结束或费用未知，暂不生成看似确定的M1预算。"]
    (output / "m0_budget.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cost_pilot(root: Path, *, live: bool = False):
    if not live:
        raise ValueError("Paid pilot requires explicit --live")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    from dotenv import load_dotenv
    load_dotenv(root / ".env")
    config = load_config(root / "configs/m0_pilot.json")
    for actor in ("agent", "user"):
        if not os.environ.get(config[actor]["api_key_env"]):
            raise ValueError(f"Missing {config[actor]['api_key_env']}")
    verify_base(root / "vendor/tau-bench", root / "patches/tau_user_simulator.patch")
    manifest = json.loads((root / "configs/task_manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if len(manifest["pilot"]) != 5:
        raise ValueError("M0 pilot requires exactly five development tasks")
    import litellm
    import tau_bench.agents.tool_calling_agent as agent_module
    import tau_bench.envs.user as user_module
    from tau_bench.envs import get_env
    output = root / "docs/results" / datetime.now().strftime("pilot_%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    environment = verify_base(root / "vendor/tau-bench", root / "patches/tau_user_simulator.patch")
    environment["litellm_version"] = importlib.metadata.version("litellm")
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    recorder, runs = CallRecorder(output / "calls.jsonl", config), []
    agent_complete, user_complete = agent_module.completion, user_module.completion
    with patch.object(agent_module, "completion", lambda **kw: recorder.call(agent_complete, "agent", **kw)), \
         patch.object(user_module, "completion", lambda **kw: recorder.call(user_complete, "user", **kw)):
        for task in manifest["pilot"]:
            recorder.task, recorder.phase = task_key(task), "initialization"
            row = {"task": recorder.task, "status": "running", "reward": None}
            print(f"Starting {recorder.task}", flush=True)
            try:
                env = get_env(task["domain"], "llm", config["user"]["model"], task["split"],
                    config["user"]["provider"], task["index"], user_temperature=config["user"]["temperature"])
                recorder.phase = "task"
                agent = agent_module.ToolCallingAgent(env.tools_info, env.wiki,
                    config["agent"]["model"], config["agent"]["provider"], config["agent"]["temperature"])
                result = agent.solve(env, task["index"], max_num_steps=config["max_steps"])
                row.update(status="finished", reward=result.reward)
            except (litellm.APIError, json.JSONDecodeError, RuntimeError) as error:
                recorder.record_failure(error)
                row.update(status="stopped", error_type=type(error).__name__, error=str(error))
            runs.append(row)
            (output / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
            write_budget(output, recorder, runs)
            print(f"{row['task']}: {row['status']}; usage records={len(recorder.records)}", flush=True)
            if row["status"] != "finished":
                break
    print(f"Pilot artifacts: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "pilot"])
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    prepare(ROOT) if args.command == "prepare" else run_cost_pilot(ROOT, live=args.live)
