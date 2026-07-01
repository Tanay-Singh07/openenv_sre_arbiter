"""
Cloud SRE Arbiter — Inference / Evaluation Script
===================================================
Drives the Nemotron-120B model through all three tasks (easy, medium, hard)
via the HTTP API to prove the environment works end-to-end.

Required environment variables:
  NEMOTRON_API_KEY  — Partner Endpoint API key (CoreWeave, DigitalOcean, etc.)
  NEMOTRON_BASE_URL — Partner Endpoint base URL (default: NVIDIA build API)
  MODEL_NAME        — Model to use (default: nvidia/nemotron-3-super-120b-a12b)
"""

import os
import re
import sys
import json
import time
import requests
from openai import OpenAI


# ---------------------------------------------------------------------------
# CONFIG (strictly from env vars — no hardcoded keys)
# ---------------------------------------------------------------------------

ENV_API_URL = os.getenv("ENV_API_URL", "http://localhost:7860")
NEMOTRON_BASE_URL = os.getenv("API_BASE_URL") or os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)
MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")
NEMOTRON_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("HF_TOKEN")
    or os.getenv("NVIDIA_API_KEY")
    or ""
)

if not NEMOTRON_API_KEY:
    print("ERROR: Set NEMOTRON_API_KEY (or NVIDIA_API_KEY) environment variable.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# SYSTEM PROMPT — tuned for Nemotron-120B strict JSON compliance
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a Tier-1 Site Reliability Engineer (SRE) responding to a live production incident.

## Mission
Every turn you MUST make two decisions:
1. **Containment:** An immediate ops action to keep the system online.
2. **Investigation:** A diagnostic query to gather root-cause evidence.

## Rules
- Do NOT guess the root cause until investigation results provide strong evidence.
- Set declare_root_cause to "unknown" while still investigating.
- Once evidence is conclusive, declare the root cause to resolve the incident.
- Every action costs money. Unnecessary spending lowers your score.
- Inaction causes system health to degrade. The system can crash.

## Strategy
Turn 1-2: Stabilize with containment AND run investigation queries.
Turn 3+: Declare root cause once evidence supports it.
Never declare a root cause on turn 1 unless evidence is absolutely conclusive.

## RESPONSE FORMAT — MANDATORY
OUTPUT ONLY A SINGLE RAW JSON OBJECT. NOTHING ELSE.
- NO markdown fences (no ```json, no ```)
- NO conversational text before or after the JSON
- NO comments inside the JSON
- NO trailing commas
- The ENTIRE response must parse as valid JSON

The JSON object MUST contain exactly these four keys:
{
  "containment_action": "scale_up_nodes | rate_limit_all | rollback_last_deploy | do_nothing",
  "investigation_query": "analyze_ip_traffic | query_db_locks | check_commit_diffs | check_service_mesh | check_resource_utilization | none",
  "declare_root_cause": "ddos_attack | viral_traffic | bad_code | database_lock | unknown",
  "justification": "1-3 sentence explanation citing specific evidence gathered"
}

VIOLATION: Any text outside the JSON object is a protocol violation and will crash the system."""


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def format_observation(obs: dict) -> str:
    """Convert an observation dict into a structured prompt for the LLM."""
    lines = [
        f"## Incident: {obs['incident_id']} (Severity: {obs['severity']})",
        f"**Situation:** {obs['initial_observation']}",
        "",
        f"### Active Alerts ({len(obs['active_alerts'])})",
    ]
    for alert in obs["active_alerts"]:
        lines.append(f"  - {alert}")

    lines.append("\n### System Metrics")
    for k, v in obs["system_metrics"].items():
        lines.append(f"  - {k}: {v}")

    lines.append(f"\n### System Health: {obs['system_health']}% | Budget Spent: ${obs['budget_spent']}")
    lines.append(f"### Turn: {obs['turn_number']} | Turns Remaining: {obs['turns_remaining']}")

    if obs.get("timeline"):
        lines.append("\n### Timeline")
        for event in obs["timeline"]:
            lines.append(f"  - {event}")

    if obs.get("investigation_results"):
        lines.append("\n### Investigation Results (Evidence Gathered)")
        for query, result in obs["investigation_results"].items():
            lines.append(f"  **{query}:** {result}")
    else:
        lines.append("\n### Investigation Results: None yet — run queries to gather evidence!")#Copyright (c) 2026 Tanay Kumar Singh (@Escanor925). All Rights Reserved

    return "\n".join(lines)


def call_env_reset(task_name: str) -> dict:
    """POST /reset to the environment API."""
    resp = requests.post(
        f"{ENV_API_URL}/reset",
        json={"task_name": task_name},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def call_env_step(action: dict) -> dict:
    """POST /step to the environment API."""
    resp = requests.post(
        f"{ENV_API_URL}/step",
        json=action,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


_SAFE_FALLBACK: dict = {
    "containment_action": "do_nothing",
    "investigation_query": "none",
    "declare_root_cause": "unknown",
    "justification": "JSON parse failed",
}


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json / ``` wrappers and any surrounding prose."""
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def parse_llm_action(content: str) -> dict:
    """
    Extract a JSON action from the LLM response.
    Strips markdown fences, isolates the JSON object, and returns a safe
    fallback dict if parsing is unrecoverable.
    """
    text = _strip_markdown_fences((content or "").strip())

    # Isolate the outermost { ... } if surrounded by conversational text
    brace_start = text.find("{")
    if brace_start != -1:
        brace_end = text.rfind("}")
        if brace_end > brace_start:
            text = text[brace_start : brace_end + 1]

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    print(f"  [PARSE] JSON unrecoverable, using safe fallback. Raw ({len(content)} chars): {content[:200]}")
    return dict(_SAFE_FALLBACK)


# ---------------------------------------------------------------------------
# MAIN EVALUATION LOOP
# ---------------------------------------------------------------------------

def run_evaluation():
    """Drive the LLM through all three tasks and report scores."""
    client = OpenAI(api_key=NEMOTRON_API_KEY, base_url=NEMOTRON_BASE_URL)
    tasks = ["easy", "medium", "hard"]
    results = {}

    print("=" * 70)
    print("  CLOUD SRE ARBITER — EVALUATION RUN")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Endpoint: {NEMOTRON_BASE_URL}")
    print(f"  Environment: {ENV_API_URL}")
    print("=" * 70)

    for task in tasks:
        print(f"\n{'─' * 70}")
        print(f"  TASK: {task.upper()}")
        print(f"{'─' * 70}")

        reset_data = call_env_reset(task)
        obs = reset_data["observation"]
        done = False

        print(f"[START] task={task}", flush=True)

        while not done:
            user_msg = format_observation(obs)

            # Stateless: system prompt + current observation only.
            # The observation already carries all accumulated state.
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]

            print(f"\n  Turn {obs['turn_number']} | Health: {obs['system_health']}% | Budget: ${obs['budget_spent']}")

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1024,
                )
                raw_content = response.choices[0].message.content
                if not raw_content:
                    raise ValueError(
                        f"LLM returned empty content. Finish reason: {response.choices[0].finish_reason}"
                    )
                action_data = parse_llm_action(raw_content)

                required_fields = [
                    "containment_action",
                    "investigation_query",
                    "declare_root_cause",
                    "justification",
                ]
                for field in required_fields:
                    if field not in action_data:
                        raise ValueError(f"Missing required field: {field}")

                print(f"  > Contain:     {action_data['containment_action']}")
                print(f"  > Investigate: {action_data['investigation_query']}")
                print(f"  > Root Cause:  {action_data['declare_root_cause']}")
                print(f"  > Reason:      {action_data['justification'][:80]}")

            except Exception as exc:
                print(f"  WARNING: LLM error: {exc}")
                action_data = {
                    "containment_action": "do_nothing",
                    "investigation_query": "check_commit_diffs",
                    "declare_root_cause": "unknown",
                    "justification": f"LLM error fallback: {exc}",
                }

            try:
                step_data = call_env_step(action_data)
                done = step_data["done"]

                if done:
                    reward = step_data["reward"]
                    score = max(0.001, min(0.999, float(reward["total_score"])))
                    results[task] = score
                    print(f"[STEP] step={obs['turn_number']} reward={score}", flush=True)
                    print(f"[END] task={task} score={score} steps={obs['turn_number']}", flush=True)
                    print(f"\n  RESOLVED — Score: {score}")
                    print(f"  Breakdown:")
                    for k, v in reward["breakdown"].items():
                        print(f"     {k}: {v}")
                else:
                    print(f"[STEP] step={obs['turn_number']} reward=0.0", flush=True)
                    obs = step_data["observation"]

            except Exception as exc:
                print(f"[STEP] step={obs['turn_number']} reward=0.001", flush=True)
                print(f"[END] task={task} score=0.001 steps={obs['turn_number']}", flush=True)
                print(f"  ERROR: Environment step failed: {exc}")
                results[task] = 0.001
                done = True

        time.sleep(0.5)

    # --- FINAL SUMMARY ---
    print(f"\n{'=' * 70}")
    print("  FINAL RESULTS")
    print(f"{'=' * 70}")
    total = 0.0
    for task in tasks:
        score = results.get(task, 0.0)
        total += score
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task.upper():8s}  {bar}  {score:.2f}")

    avg = total / len(tasks) if tasks else 0.0
    print(f"{'─' * 70}")
    print(f"  AVERAGE SCORE: {avg:.2f}")
    print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    run_evaluation()
