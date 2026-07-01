"""
Cloud SRE Arbiter — FastAPI Server
====================================
Serves the OpenEnv-compliant HTTP API with /reset, /step, and /state
endpoints.  The GET / health-check is required by automated judges.
"""

import os
import re
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
from openai import OpenAI

from environment import (
    CloudSREEnv,
    Action,
)

# ---------------------------------------------------------------------------
# LLM JSON SANITIZER
# ---------------------------------------------------------------------------

# Fallback values when the LLM returns an invalid enum value
_VALID_CONTAINMENT = {"scale_up_nodes", "rate_limit_all", "rollback_last_deploy", "do_nothing"}
_VALID_INVESTIGATION = {"analyze_ip_traffic", "query_db_locks", "check_commit_diffs", "check_service_mesh", "check_resource_utilization", "none"}
_VALID_ROOT_CAUSE = {"ddos_attack", "viral_traffic", "bad_code", "database_lock", "unknown"}


def clean_llm_json(raw_text: str) -> dict:
    """
    Extract a JSON object from raw LLM output that may contain markdown
    fences, conversational filler, or truncation artifacts.

    Returns a sanitized dict with guaranteed valid Action enum values,
    including a truncation-healer fallback for broken justification fields.
    """
    text = (raw_text or "").strip()

    if not text:
        raise ValueError("LLM returned empty content")

    # 1) Strip markdown fences if present.
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # 2) Isolate probable JSON object boundaries.
    brace_start = text.find("{")
    if brace_start != -1:
        brace_end = text.rfind("}")
        text = text[brace_start : brace_end + 1] if brace_end > brace_start else text[brace_start:]

    def _extract_field(source: str, key: str, default: str) -> str:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', source)#Copyright (c) 2026 Tanay Kumar Singh (@Escanor925). All Rights Reserved
        if not match:
            return default
        value = match.group(1).strip()
        return value if value else default

    # 3) First parse attempt.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

        # 4) Healer fallback: if justification is truncated, replace it with safe text.
        marker = '"justification"'
        marker_index = text.find(marker)
        if marker_index != -1:
            prefix = text[:marker_index].rstrip()
            if prefix.endswith(","):
                prefix = prefix[:-1].rstrip()
            if "{" in prefix:
                prefix = prefix[prefix.find("{"):]
            else:
                prefix = "{"

            healed_text = f'{prefix}{"" if prefix.endswith("{") else ", "}"justification": "Truncated by API constraints."}}'
            try:
                parsed = json.loads(healed_text)
                print("[AUTOPILOT] Applied JSON healer fallback for truncated justification field.")
            except json.JSONDecodeError:
                parsed = None

        # 5) Last-resort extraction so Auto-Pilot does not hard-crash on malformed text.
        if parsed is None:
            parsed = {
                "containment_action": _extract_field(text, "containment_action", "do_nothing"),
                "investigation_query": _extract_field(text, "investigation_query", "none"),
                "declare_root_cause": _extract_field(text, "declare_root_cause", "unknown"),
                "justification": "Truncated by API constraints.",
            }
            print("[AUTOPILOT] JSON parse failed; using regex extraction fallback.")

    if not isinstance(parsed, dict):
        print(f"[AUTOPILOT] LLM returned non-object JSON: {type(parsed)}")
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

    # 4. Sanitize enum values — fall back to safe defaults if LLM hallucinated
    containment = parsed.get("containment_action", "do_nothing")
    investigation = parsed.get("investigation_query", "none")
    root_cause = parsed.get("declare_root_cause", "unknown")
    justification = parsed.get("justification", "")

    return {
        "containment_action": containment if containment in _VALID_CONTAINMENT else "do_nothing",
        "investigation_query": investigation if investigation in _VALID_INVESTIGATION else "none",
        "declare_root_cause": root_cause if root_cause in _VALID_ROOT_CAUSE else "unknown",
        "justification": justification.strip() if isinstance(justification, str) and justification.strip() else "AI agent could not provide justification.",
    }


# ---------------------------------------------------------------------------
# APP SETUP
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenEnv — Cloud SRE Arbiter",
    description=(
        "A multi-step, RL-style environment testing an AI agent's ability to "#Copyright (c) 2026 Tanay Kumar Singh (@Escanor925). All Rights Reserved
        "balance system-uptime containment with root-cause investigation "
        "under severe financial constraints."
    ),
    version="1.0.0",
)

_default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:7860",
    "http://127.0.0.1:7860",
]
_space_host = os.environ.get("SPACE_HOST")
if _space_host:
    _default_origins.append(f"https://{_space_host}")

_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
_allowed_origins = [origin.strip() for origin in _allowed_origins_env.split(",") if origin.strip()]
if not _allowed_origins:
    _allowed_origins = _default_origins

_allowed_autopilot_base_url = os.environ.get("API_BASE_URL") or os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
_allowed_autopilot_model = os.environ.get("MODEL_NAME") or os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the environment engine (loads data.json once at startup)
try:
    env = CloudSREEnv(data_path="data.json")
except FileNotFoundError:
    print("WARNING: data.json not found at startup — will retry on first request.")
    env = CloudSREEnv.__new__(CloudSREEnv)   # placeholder; will fail gracefully


# ---------------------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = Field(
        "easy",
        description="Difficulty level: easy | medium | hard",
    )


class ResetResponse(BaseModel):
    status: int = 200
    observation: dict
    state: dict


class StepResponse(BaseModel):
    status: int = 200
    observation: Optional[dict] = None
    reward: dict
    done: bool
    info: dict
    state: dict


class AutoPilotRequest(BaseModel):
    model: str = Field("nvidia/nemotron-3-super-120b-a12b", min_length=1)
    base_url: str = Field("https://integrate.api.nvidia.com/v1", min_length=1)
    messages: list[dict]
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1, le=4096)


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def root_health():
    """Root health check for automated judge pings."""
    return {"status": "ok"}


@app.get("/health", tags=["health"])
def health_check():
    """Automated judges ping this to verify the container is alive."""
    return {
        "status": "ok",
        "environment": "Cloud SRE Arbiter",
        "version": "1.0.0",
    }


@app.post("/reset", response_model=ResetResponse, tags=["environment"])
def reset_env(req: Optional[ResetRequest] = None):
    """
    Initialize a new episode for the given task difficulty and return the
    first Observation the agent will see.
    """
    try:
        task_name = req.task_name if req is not None else "easy"
        obs = env.reset(task_name)
        return ResetResponse(
            status=200,
            observation=obs.model_dump(),
            state=env.get_state().model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset failed: {exc}")


@app.post("/step", response_model=StepResponse, tags=["environment"])
def step_env(action: Action):
    """
    Process the agent's Action (containment + investigation + declaration),
    update internal state, and return (Observation, Reward, done, info).
    """
    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(
            status=200,
            observation=obs.model_dump() if obs else None,
            reward=reward.model_dump(),
            done=done,
            info=info,
            state=env.get_state().model_dump(),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Step failed: {exc}")


@app.get("/state", tags=["environment"])
def get_state():
    """Return current episode metadata."""
    try:
        return env.get_state().model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"State query failed: {exc}")


@app.post("/autopilot", tags=["llm"])
def autopilot(req: AutoPilotRequest):
    """
    Server-side LLM proxy for the dashboard Auto-Pilot flow.
    Calls the LLM, sanitizes the JSON response, validates against
    the Action schema, and returns a clean action dict to the frontend.
    """
    # 1. Call the LLM
    try:
        if req.base_url != _allowed_autopilot_base_url:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported base_url. Use {_allowed_autopilot_base_url}.",
            )

        if req.model != _allowed_autopilot_model:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model. Use {_allowed_autopilot_model}.",
            )

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="Server misconfiguration: no API key found. Set OPENAI_API_KEY, HF_TOKEN, or NVIDIA_API_KEY.",
            )

        client = OpenAI(api_key=api_key, base_url=req.base_url)
        truncation_guard_prompt = (
            "CRITICAL: YOUR OUTPUT IS BEING TRUNCATED BY SERVER LIMITS. "
            "THE 'justification' FIELD MUST BE EXTREMELY BRIEF. MAXIMUM 10 WORDS. "
            "DO NOT WRITE LONG SENTENCES OR YOUR OUTPUT WILL CORRUPT."
        )
        llm_messages = [{"role": "system", "content": truncation_guard_prompt}, *req.messages]
        response = client.chat.completions.create(
            model=req.model,
            messages=llm_messages,
            temperature=req.temperature,
            max_tokens=max(1500, req.max_tokens),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM upstream call failed: {exc}")

    # 2. Extract raw content and inspect finish_reason safety net
    try:
        choice = response.choices[0]
        raw_content = choice.message.content or ""
        finish_reason = (choice.finish_reason or "").lower()
    except (IndexError, AttributeError, TypeError) as exc:
        raise HTTPException(status_code=502, detail=f"Malformed LLM response: {exc}")

    if finish_reason == "length":
        print("\n" + "=" * 96)
        print("CRITICAL: LLM OUTPUT WAS TRUNCATED DUE TO TOKEN LIMITS! (finish_reason=length)")
        print("CRITICAL: Auto-Pilot aborted before JSON parsing to prevent invalid-action execution.")
        print("=" * 96 + "\n")
        raise HTTPException(
            status_code=500,
            detail=(
                "Auto-Pilot failed: model output was truncated because it ran out of tokens "
                "(finish_reason=length). Increase token budget and retry."
            ),
        )

    print(f"[AUTOPILOT] Raw LLM response ({len(raw_content)} chars): {raw_content[:300]}")

    # 3. Sanitize and parse JSON
    try:
        action = clean_llm_json(raw_content)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"LLM JSON parsing failed: {exc}")

    # 4. Return the cleaned action as a JSON content envelope
    #    (frontend expects { content: "<json string>" })
    return {"content": json.dumps(action)}


# ---------------------------------------------------------------------------
# FRONTEND UI ROUTING
# ---------------------------------------------------------------------------

_ui_dir = os.path.join(os.path.dirname(__file__), "dist")
if os.path.exists(_ui_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(_ui_dir, "assets")), name="assets")
    _ui_root = os.path.abspath(_ui_dir)
    _post_only_endpoints = {"reset", "step", "autopilot"}

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        if full_path in _post_only_endpoints:
            raise HTTPException(status_code=405, detail="Method Not Allowed")

        normalized = os.path.normpath(full_path).lstrip("\\/")
        path = os.path.abspath(os.path.join(_ui_root, normalized))

        if path != _ui_root and not path.startswith(_ui_root + os.sep):
            return FileResponse(os.path.join(_ui_dir, "index.html"))

        return FileResponse(path) if os.path.exists(path) and os.path.isfile(path) else FileResponse(os.path.join(_ui_dir, "index.html"))
