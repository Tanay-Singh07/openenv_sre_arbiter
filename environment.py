"""
Cloud SRE Arbiter — Core Environment Engine
============================================
Implements a multi-step, RL-style environment where an AI agent must
simultaneously contain live incidents (Ops) and investigate root causes
(Sec/Data). The grader is fully deterministic and penalizes reckless
actions such as guessing the root cause without gathering evidence first.
"""

import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, Any, Tuple, Optional, Literal, List


# ---------------------------------------------------------------------------
# 1. PYDANTIC MODELS  (strict schemas required by OpenEnv)
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees each turn."""
    incident_id: str = Field(..., description="Unique incident identifier")
    severity: str = Field(..., description="Incident severity level (P1/P2/P3)")
    initial_observation: str = Field(..., description="Human-readable summary of what is happening")
    active_alerts: List[str] = Field(..., description="List of active alert names")
    system_metrics: Dict[str, str] = Field(..., description="Current system metric readings")
    timeline: List[str] = Field(..., description="Recent event timeline")
    investigation_results: Dict[str, str] = Field(
        default_factory=dict,
        description="Results from investigation queries run so far"
    )
    system_health: float = Field(..., ge=0.0, le=100.0, description="Current system health 0-100")
    budget_spent: float = Field(..., ge=0.0, description="Total budget consumed so far ($)")
    turn_number: int = Field(..., ge=0, description="Current turn in this episode")
    turns_remaining: int = Field(..., ge=0, description="Turns left before forced resolution")
    available_actions: Dict[str, List[str]] = Field(
        ..., description="Available action choices for each action type"
    )


class Action(BaseModel):
    """The agent's two-pronged decision each turn."""
    containment_action: Literal[
        "scale_up_nodes",
        "rate_limit_all",
        "rollback_last_deploy",
        "do_nothing"
    ] = Field(..., description="Immediate ops action to keep the system online")#Copyright (c) 2026 Tanay Kumar Singh (@Tanay-Singh07). All Rights Reserved

    investigation_query: Literal[
        "analyze_ip_traffic",
        "query_db_locks",
        "check_commit_diffs",
        "check_service_mesh",
        "check_resource_utilization",
        "none"
    ] = Field(..., description="Query to run for root-cause investigation")

    declare_root_cause: Literal[
        "ddos_attack",
        "viral_traffic",
        "bad_code",
        "database_lock",
        "unknown"
    ] = Field(..., description="Declare the root cause or 'unknown' to keep investigating")

    justification: str = Field(
        ...,
        min_length=1,
        description="A short explanation for this decision, citing evidence gathered"
    )


class Reward(BaseModel):
    """Deterministic grading result returned after each step."""
    total_score: float = Field(..., gt=0.0, lt=1.0, description="Final score in (0, 1) exclusive")
    breakdown: Dict[str, float] = Field(..., description="Score breakdown by category")


class State(BaseModel):
    """Metadata about the current episode."""
    task_name: str = Field(..., description="Current task difficulty level")
    incident_id: str = Field("", description="Current incident ID")
    turn_number: int = Field(0, description="Current turn")
    max_turns: int = Field(0, description="Maximum turns allowed")
    system_health: float = Field(100.0, description="Current system health")
    budget_spent: float = Field(0.0, description="Budget consumed")
    is_done: bool = Field(False, description="Whether the episode has ended")


# ---------------------------------------------------------------------------
# 2. COST & REWARD CONSTANTS
# ---------------------------------------------------------------------------

# Containment costs (represent real infrastructure spend)
CONTAINMENT_COSTS = {
    "scale_up_nodes": 500.0,
    "rate_limit_all": 100.0,
    "rollback_last_deploy": 200.0,
    "do_nothing": 0.0,
}

# How each containment action affects system health (additive per turn)
CONTAINMENT_HEALTH_EFFECTS = {
    "scale_up_nodes": +15.0,
    "rate_limit_all": +10.0,
    "rollback_last_deploy": +20.0,
    "do_nothing": -15.0,          # doing nothing while system is on fire is bad
}

# Investigation costs
INVESTIGATION_COST = 50.0        # each query costs $50

# Reward weights (must sum to 1.0)
W_ROOT_CAUSE = 0.40               # correctly identifying the root cause
W_CONTAINMENT = 0.25              # picking the ideal containment action
W_EVIDENCE = 0.15                 # gathering required evidence before declaring
W_EFFICIENCY = 0.10               # budget efficiency
W_HEALTH = 0.10                   # keeping system health above critical

# Penalties
PREMATURE_GUESS_PENALTY = 0.30    # deducted if you declare root cause without evidence
SYSTEM_CRASH_PENALTY = 0.50       # deducted if system health drops to 0
MAX_BUDGET = 5000.0               # budget ceiling for efficiency calculation
MAX_TURNS = 6                     # maximum turns per incident


# ---------------------------------------------------------------------------
# 3. ENVIRONMENT ENGINE
# ---------------------------------------------------------------------------

class CloudSREEnv:
    """
    Gymnasium-style environment for the Cloud SRE Arbiter.

    The agent loops through reset() -> step() -> step() -> ... until done.
    Each task (easy/medium/hard) contains one incident scenario.
    """

    def __init__(self, data_path: str = "data.json"):
        # Try loading from the same directory as this file first
        p = Path(__file__).parent / data_path
        if not p.exists():
            p = Path(data_path)
        with open(p, "r", encoding="utf-8") as f:
            self.dataset: Dict[str, list] = json.load(f)#Copyright (c) 2026 Tanay Kumar Singh (@Tanay-Singh07). All Rights Reserved

        # Episode state
        self._task_name: str = ""
        self._case: Optional[dict] = None
        self._turn: int = 0
        self._budget: float = 0.0
        self._health: float = 50.0       # start at 50 — system is already degraded
        self._investigation_results: Dict[str, str] = {}
        self._evidence_gathered: List[str] = []
        self._containment_used: List[str] = []
        self._done: bool = True

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def reset(self, task_name: str = "easy") -> Observation:
        """Start a new episode for the given task difficulty."""
        if task_name not in self.dataset:
            raise ValueError(
                f"Task '{task_name}' not found. Available: {list(self.dataset.keys())}"
            )

        self._task_name = task_name
        self._case = self.dataset[task_name][0]   # one case per difficulty
        self._turn = 0
        self._budget = 0.0
        self._health = 50.0                        # system is already hurting
        self._investigation_results = {}
        self._evidence_gathered = []
        self._containment_used = []
        self._done = False

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Optional[Observation], Reward, bool, Dict[str, Any]]:
        """
        Process one agent turn.

        Returns: (observation, reward, done, info)
        """
        if self._done or self._case is None:
            raise RuntimeError("Episode is over. Call reset() to start a new one.")

        self._turn += 1
        ground_truth = self._case["ground_truth"]
        hidden_data = self._case["hidden_data"]
        info: Dict[str, Any] = {"justification": action.justification, "turn": self._turn}

        # --- A) PROCESS CONTAINMENT ---
        cost = CONTAINMENT_COSTS.get(action.containment_action, 0.0)
        self._budget += cost
        health_delta = CONTAINMENT_HEALTH_EFFECTS.get(action.containment_action, 0.0)
        self._health = max(0.0, min(100.0, self._health + health_delta))
        if action.containment_action != "do_nothing":
            self._containment_used.append(action.containment_action)

        # --- B) PROCESS INVESTIGATION ---
        if action.investigation_query != "none":
            self._budget += INVESTIGATION_COST
            query = action.investigation_query
            if query in hidden_data:
                self._investigation_results[query] = hidden_data[query]
            else:
                self._investigation_results[query] = "Query returned no anomalies."
            if query not in self._evidence_gathered:
                self._evidence_gathered.append(query)

        # --- C) CHECK END CONDITIONS ---
        declared = action.declare_root_cause != "unknown"
        timed_out = self._turn >= MAX_TURNS
        system_crashed = self._health <= 0.0

        if declared or timed_out or system_crashed:
            self._done = True
            reward = self._grade(action, ground_truth, timed_out, system_crashed)
            info["grading_detail"] = reward.breakdown
            return None, reward, True, info

        # --- D) CONTINUE INVESTIGATING ---
        # Natural health decay each turn (the incident is ongoing)
        self._health = max(0.0, self._health - 5.0)

        reward = Reward(
            total_score=0.001,
            breakdown={
                "status": 0.0,
                "message_investigating": 0.0,
                "budget_spent": self._budget,
                "system_health": self._health,
            },
        )
        return self._build_observation(), reward, False, info

    def get_state(self) -> State:
        """Return metadata about the current episode."""
        return State(
            task_name=self._task_name or "none",
            incident_id=self._case["incident_id"] if self._case else "",
            turn_number=self._turn,
            max_turns=MAX_TURNS,
            system_health=self._health,
            budget_spent=self._budget,
            is_done=self._done,
        )

    # ------------------------------------------------------------------
    # DETERMINISTIC GRADER
    # ------------------------------------------------------------------

    def _grade(
        self,
        action: Action,
        ground_truth: dict,
        timed_out: bool,
        system_crashed: bool,
    ) -> Reward:
        """
        Score the agent's performance.  Returns a float in [0.0, 1.0].

        Scoring breakdown:
          - Root cause identification    (40%)
          - Containment quality          (25%)
          - Evidence gathering           (15%)
          - Budget efficiency            (10%)
          - System health maintenance    (10%)

        Penalties:
          - Premature guess (no evidence)  → −0.30
          - System crash (health → 0)      → −0.50
        """
        breakdown: Dict[str, float] = {}

        # 1. Root cause (40%)
        if action.declare_root_cause == ground_truth["root_cause"]:
            breakdown["root_cause"] = W_ROOT_CAUSE
        elif timed_out and action.declare_root_cause == "unknown":
            breakdown["root_cause"] = 0.0   # never even guessed
        else:
            breakdown["root_cause"] = 0.0   # wrong guess

        # 2. Containment (25%) — check if ideal action was used at any point
        if ground_truth["ideal_containment"] in self._containment_used:
            breakdown["containment"] = W_CONTAINMENT
        elif action.containment_action == ground_truth["ideal_containment"]:
            breakdown["containment"] = W_CONTAINMENT
        else:
            breakdown["containment"] = 0.0

        # 3. Evidence (15%) — did the agent gather the required evidence?
        required = set(ground_truth.get("required_evidence", []))
        gathered = set(self._evidence_gathered)
        if required and required.issubset(gathered):
            breakdown["evidence"] = W_EVIDENCE
        elif required:
            # Partial credit for gathering some evidence
            overlap = len(required & gathered) / len(required)
            breakdown["evidence"] = round(W_EVIDENCE * overlap, 4)
        else:
            breakdown["evidence"] = W_EVIDENCE  # no evidence required

        # 4. Budget efficiency (10%)
        if self._budget <= 0:
            breakdown["efficiency"] = W_EFFICIENCY
        else:
            breakdown["efficiency"] = round(
                max(0.0, W_EFFICIENCY * (1.0 - self._budget / MAX_BUDGET)), 4
            )

        # 5. System health (10%)
        breakdown["health"] = round(W_HEALTH * (self._health / 100.0), 4)

        # --- Penalties ---
        penalty = 0.0

        # Premature guess: agent declared root cause without gathering any
        # of the required evidence
        if (
            action.declare_root_cause != "unknown"
            and required
            and not required.issubset(gathered)
        ):
            penalty += PREMATURE_GUESS_PENALTY
            breakdown["penalty_premature_guess"] = -PREMATURE_GUESS_PENALTY

        # System crash penalty
        if system_crashed:
            penalty += SYSTEM_CRASH_PENALTY
            breakdown["penalty_system_crash"] = -SYSTEM_CRASH_PENALTY

        raw = sum(v for k, v in breakdown.items() if not k.startswith("penalty_"))
        total = max(0.001, min(0.999, round(raw - penalty, 4)))

        breakdown["budget_spent"] = self._budget
        breakdown["final_health"] = self._health
        breakdown["turns_used"] = float(self._turn)

        return Reward(total_score=total, breakdown=breakdown)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """Build an Observation from the current case + internal state."""
        case = self._case
        if case is None:
            raise RuntimeError("No active case — call reset() first.")

        return Observation(
            incident_id=case["incident_id"],
            severity=case.get("severity", "P1"),
            initial_observation=case.get("initial_observation", ""),
            active_alerts=case["active_alerts"],
            system_metrics=case["system_metrics"],
            timeline=case.get("timeline", []),
            investigation_results=dict(self._investigation_results),
            system_health=round(self._health, 2),
            budget_spent=round(self._budget, 2),
            turn_number=self._turn,
            turns_remaining=MAX_TURNS - self._turn,
            available_actions={
                "containment_action": [
                    "scale_up_nodes", "rate_limit_all",
                    "rollback_last_deploy", "do_nothing",
                ],
                "investigation_query": [
                    "analyze_ip_traffic", "query_db_locks",
                    "check_commit_diffs", "check_service_mesh",
                    "check_resource_utilization", "none",
                ],
                "declare_root_cause": [
                    "ddos_attack", "viral_traffic",
                    "bad_code", "database_lock", "unknown",
                ],
            },
        )
