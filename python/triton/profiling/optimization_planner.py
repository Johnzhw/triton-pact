"""
PACT Optimization Planner — Bottleneck → PACT pass recommendation engine.

Core responsibilities:
  1. Receive bottleneck analysis results from OnlineProfiler
  2. Map bottleneck type → PACT pass enable/disable + parameter tuning
  3. Generate new PACT environment variable combinations
  4. Track optimization history to avoid cycles
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class OptimizationAction:
    """One optimization action: enable/disable a pass or tune a parameter."""

    action_type: str  # "enable_pass" | "disable_pass" | "set_param"
    target: str       # pass name or parameter name
    value: str        # new value

    def to_env_dict(self) -> Dict[str, str]:
        return {self.target: self.value}


class OptimizationPlanner:
    """
    Bottleneck-to-PACT-pass mapping engine.

    Input:  bottleneck type + current PACT configuration
    Output: new PACT configuration (environment variable dictionary)

    Architecture-aware: filters recommendations by SM version.
    Cycle-avoidance: tracks history to avoid re-trying failed configs.
    """

    # ═══════════════════════════════════════════════════════════
    # Bottleneck → Pass mapping table
    # ═══════════════════════════════════════════════════════════
    BOTTLENECK_PASS_MAP = {
        "low_occupancy": {
            "primary": [
                OptimizationAction(
                    "set_param", "PACT_ENABLE_AUTO_NUM_WARPS", "1"
                ),
                OptimizationAction(
                    "set_param", "PACT_MAX_PIPELINE_STAGES", "3"
                ),
                OptimizationAction(
                    "set_param", "PACT_ENABLE_AUTO_NUM_STAGES", "1"
                ),
            ],
            "secondary": [
                OptimizationAction(
                    "set_param",
                    "PACT_AXISINFO_OVERRIDE_STRATEGY",
                    "ampere_adaptive",
                ),
            ],
            "description": "Low occupancy detected → reduce num_stages, "
            "adjust num_warps, use Ampere-conservative contiguity",
        },
        "memory_bandwidth_bound": {
            "primary": [
                OptimizationAction(
                    "set_param",
                    "PACT_AXISINFO_OVERRIDE_STRATEGY",
                    "hopper_aggressive",
                ),
                OptimizationAction(
                    "set_param", "PACT_MAX_PIPELINE_STAGES", "6"
                ),
                OptimizationAction(
                    "set_param", "PACT_ENABLE_AUTO_NUM_STAGES", "1"
                ),
            ],
            "secondary": [
                OptimizationAction(
                    "set_param", "PACT_ENABLE_RUN_COALESCE", "1"
                ),
            ],
            "description": "Memory bandwidth bound → aggressive contiguity, "
            "more pipeline stages, enable coalescing",
        },
        "poor_locality": {
            "primary": [
                OptimizationAction(
                    "set_param", "PACT_ENABLE_PAGE_MAJOR_TILE", "1"
                ),
                OptimizationAction(
                    "set_param", "PACT_ENABLE_RUN_COALESCE", "1"
                ),
            ],
            "secondary": [],
            "description": "Poor L2 locality → enable page-major tile "
            "reordering and run coalescing",
        },
        "compute_bound": {
            "primary": [
                OptimizationAction(
                    "set_param", "PACT_ENABLE_DOT_PROMOTION", "1"
                ),
            ],
            "secondary": [
                OptimizationAction(
                    "set_param", "PACT_ENABLE_AUTO_NUM_WARPS", "1"
                ),
            ],
            "description": "Compute bound → enable dot promotion to MMA, "
            "tune num_warps",
        },
    }

    def __init__(
        self,
        current_env: Optional[Dict[str, str]] = None,
        sm_version: int = 86,
    ):
        self.current_env = (current_env or {}).copy()
        self.sm_version = sm_version
        self.optimization_history: List[Dict] = []

    def plan(self, bottleneck: Dict) -> Optional[Dict[str, str]]:
        """
        Generate optimization plan for detected bottleneck.

        Args:
            bottleneck: bottleneck analysis result from OnlineProfiler

        Returns:
            New environment variable dictionary, or None if no action needed
        """
        bottleneck_type = bottleneck.get("type", "")
        actions_map = self.BOTTLENECK_PASS_MAP.get(bottleneck_type, {})

        if not actions_map:
            print(
                f"[PACT Planner] Unknown bottleneck type: {bottleneck_type}"
            )
            return None

        # Try primary actions first
        new_env = self._apply_actions(
            self.current_env, actions_map.get("primary", [])
        )

        # Check for duplicate (already tried this config)
        if self._is_duplicate(new_env):
            # Fall back to secondary actions
            new_env = self._apply_actions(
                self.current_env, actions_map.get("secondary", [])
            )
            if self._is_duplicate(new_env):
                print(
                    "[PACT Planner] All actions exhausted for "
                    f"'{bottleneck_type}' — keeping current config."
                )
                return None

        # SM architecture filtering
        new_env = self._filter_by_architecture(new_env)

        # Record in history
        self.optimization_history.append(
            {
                "bottleneck": bottleneck_type,
                "description": actions_map.get("description", ""),
                "result_env": new_env,
            }
        )

        # Update current env
        self.current_env = new_env

        print(
            f"[PACT Planner] Recommendation for '{bottleneck_type}': "
            f"{self._diff_env(self.current_env)}"
        )
        return new_env

    def _apply_actions(
        self, env: Dict[str, str], actions: List[OptimizationAction]
    ) -> Dict[str, str]:
        """Apply a list of optimization actions to environment dict."""
        result = env.copy()
        for action in actions:
            result[action.target] = action.value
        return result

    def _is_duplicate(self, new_env: Dict[str, str]) -> bool:
        """Check if this config has been tried in recent history."""
        recent_keys = {"PACT_ENABLE_AUTO_NUM_STAGES",
                       "PACT_AMPERE_CONTIGUITY_CAP",
                       "PACT_MAX_PIPELINE_STAGES"}
        for h in self.optimization_history[-5:]:
            old_env = h.get("result_env", {})
            # Compare key parameters only
            for k in recent_keys:
                if old_env.get(k) != new_env.get(k):
                    break
            else:
                return True  # All key params match
        return False

    def _filter_by_architecture(
        self, env: Dict[str, str]
    ) -> Dict[str, str]:
        """Remove recommendations incompatible with current SM architecture."""
        result = env.copy()

        if self.sm_version < 90:
            # Ampere: disable page-major tile (breaks Pipeline pattern)
            if result.get("PACT_ENABLE_PAGE_MAJOR_TILE") == "1":
                print(
                    "[PACT Planner] Ampere detected — "
                    "disabling P7 (page-major tile breaks Pipeline)"
                )
                result["PACT_ENABLE_PAGE_MAJOR_TILE"] = "0"

            # Ampere: cap pipeline stages at 4
            if int(result.get("PACT_MAX_PIPELINE_STAGES", "4")) > 4:
                result["PACT_MAX_PIPELINE_STAGES"] = "4"

        return result

    def _diff_env(self, new_env: Dict[str, str]) -> str:
        """Generate human-readable diff between old and new env."""
        changes = []
        for k, v in new_env.items():
            if k.startswith("PACT_"):
                old_v = self.optimization_history[-1]["result_env"].get(
                    k, "N/A"
                ) if self.optimization_history else "N/A"
                if str(old_v) != str(v):
                    changes.append(f"{k}={v}")
        return ", ".join(changes) if changes else "no changes"

    def get_history(self) -> List[Dict]:
        """Get full optimization history for reporting."""
        return self.optimization_history

    def reset(self):
        """Reset planner state."""
        self.optimization_history.clear()
        self.current_env = {}
