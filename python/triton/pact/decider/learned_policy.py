"""Learned offline policy: JSON rule model -> variant env (PGO tree only).

Deployment contract (V9-B3):
* ``PACT_FAMILY_MODEL`` points at a JSON file produced by
  ``suite/offline/learned/learn_policy.py --export`` with one of:
    {"kind": "tree",  "rules": [...]}   — list of {feat, thr, lo, hi, pick}
    {"kind": "knn",   "prototypes": [...]} — list of {feat, gains}
  plus {"variants": {name: env-dict}} mapping model names to PACT env
  presets (theory/occ/lat/deep...).  Inference is pure Python.
* ``decide()`` consults the learned model first (when the env var is set
  and the file parses); on any error or miss it falls back to the
  FamilyTable.  ``vanilla`` remains a first-class prediction and compiles
  the untouched kernel.

Safety envelope: Lemma 4 — every variant in ``variants`` only changes
scheduling (stages/warps/PACT env); the policy therefore cannot change the
kernel's read set, only which legal schedule runs.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from triton.pact.decider.family_table import FamilyTable

FEATURE_ORDER = ("logS", "logD", "logP", "B")


def features(cfg: Dict[str, Any]) -> list:
    return [math.log2(cfg["S"]), math.log2(cfg["D"]),
            math.log2(cfg.get("P", 16)), cfg.get("B", 1)]


class LearnedPolicy:
    def __init__(self, data: Dict):
        self.kind = data.get("kind")
        self.variants = data.get("variants") or {}
        self.tree = data.get("rules") or []
        self.prototypes = data.get("prototypes") or []
        self.k = int(data.get("k", 3))
        self.max_dist = data.get("max_dist")
        self.default_pick = data.get("default", "theory")

    @classmethod
    def load(cls, path: Optional[str] = None) -> Optional["LearnedPolicy"]:
        path = path or os.environ.get("PACT_FAMILY_MODEL")
        if not path or not Path(path).is_file():
            return None
        try:
            return cls(json.loads(Path(path).read_text()))
        except Exception:
            return None

    def _predict_tree(self, x):
        node = self.tree[0] if self.tree else None
        # rules: nested via lo/hi indices into the list
        i = 0
        while node is not None and "pick" not in node:
            go_lo = x[node["feat"]] <= node["thr"]
            nxt = node.get("lo" if go_lo else "hi")
            if nxt is None:
                return "vanilla", 1.0
            i, node = nxt, self.tree[nxt] if isinstance(nxt, int) else None
        return (node or {}).get("pick", "vanilla"), 1.0

    def _predict_knn(self, x):
        import statistics
        ds = []
        for p in self.prototypes:
            d = sum(abs(a - b) for a, b in zip(x, p["feat"]))
            ds.append((d, p.get("gains", {})))
        ds.sort(key=lambda t: t[0])
        # V9: outside the offline-measured neighbourhood, fall back to the
        # model's safe default (the static floor) instead of extrapolating.
        max_dist = getattr(self, "max_dist", None)
        default = getattr(self, "default_pick", "theory")
        if max_dist is not None and ds and ds[0][0] > max_dist:
            return default, 1.0
        top = ds[: self.k]
        best, best_v = default, 1.0
        for v in top[0][1]:
            vals = [g.get(v, 1.0) for _, g in top if v in g]
            if not vals:
                continue
            m = statistics.median(vals)
            if m > best_v:
                best, best_v = v, m
        return best, best_v

    def predict(self, cfg: Dict[str, Any]) -> Tuple[str, float]:
        x = features(cfg)
        if self.kind == "tree":
            return self._predict_tree(x)
        if self.kind == "knn":
            return self._predict_knn(x)
        return "vanilla", 1.0

    def env_for(self, variant: str) -> Optional[Dict[str, str]]:
        return self.variants.get(variant)


def decide(facts: Dict[str, Any], batch: int, seq_len: int,
           table: Optional[FamilyTable] = None,
           hints_dir: Optional[Path] = None,
           head_dim: Optional[int] = None,
           gqa: Optional[int] = None,
           cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Same contract as online_decider.decide, with the learned policy first.

    The caller (service_main) may pass the kernel geometry ``cfg``
    ({D,P,S,B,Hq,GQA}); without it the learned policy is skipped and the
    FamilyTable path behaves exactly as before.

    V10-P0 (BEH-2): when ``cfg`` carries D/GQA they also address the
    ``d{D}g{gqa}|``-prefixed FamilyTable entries on the table fallback path
    (explicit head_dim/gqa arguments still win).  Previously the service
    path looked the plain keys up only, so prefixed entries were
    unaddressable there (the v7 dead-key lesson in residual form).
    """
    from triton.pact.decider.online_decider import decide as table_decide
    if cfg is None:
        return table_decide(facts, batch, seq_len, table=table,
                            hints_dir=hints_dir, head_dim=head_dim, gqa=gqa)
    if head_dim is None and cfg.get("D"):
        head_dim = int(cfg["D"])
    if gqa is None and cfg.get("GQA"):
        gqa = int(cfg["GQA"])
    policy = LearnedPolicy.load()
    if policy is not None:
        try:
            variant, _ = policy.predict(cfg)
            env = policy.env_for(variant)
            if env is not None:
                if variant == "vanilla":
                    return {"family": "vanilla",
                            "extra_env": {"PACT_ENABLE": "0"},
                            "options_override": {}, "hints": {}}
                out = table_decide(facts, batch, seq_len, table=table,
                                   hints_dir=hints_dir, head_dim=head_dim,
                                   gqa=gqa)
                out["family"] = variant
                out["extra_env"] = dict(env)
                return out
        except Exception:
            pass
    return table_decide(facts, batch, seq_len, table=table, hints_dir=hints_dir,
                        head_dim=head_dim, gqa=gqa)
