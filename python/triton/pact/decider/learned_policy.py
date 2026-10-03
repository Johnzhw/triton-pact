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

# V17 S1-3 ②: every previously-silent exception branch counts here and
# keeps the last message; the JSON sinks surface it across processes.
_SWALLOWED: Dict[str, Any] = {"predict_errors": 0, "load_errors": 0,
                              "schema_refused": 0, "last_error": None}

# V21-D (F14): FEATURE_ORDER was a dead 4-wide constant left from the
# pre-T6 era (features() returns 7 since V16-T6 and nothing read it);
# removed.  The runtime-truth extension lives in features11 below.

_RUNTIME_KEYS = ("stall_memory_permille", "l2_hit_permille",
                 "active_warp_ratio_permille", "sm_efficiency_permille")


def features(cfg: Dict[str, Any]) -> list:
    # V16-T6 learning upgrade: add two generalization features --
    # GQA ratio and the STATIC KV working-set proxy log2(S*D*Hk*2*2)
    # (bytes of one sequence's K+V in bf16) -- so a fitted winner table
    # can separate hot/cold regimes the way S1's residency gate does
    # without any runtime counter.  Older 4-feature models still load;
    # their prototypes simply have shorter vectors and knn distance is
    # computed with zip() truncation (documented compatibility note).
    # V18 B3 v7: 7th feature = the GQA x D interaction log2(GQA*D) --
    # per-KV-head served load (query heads sharing one kv head times
    # head dim), the axis the V18 24-cell grid showed the winner flips
    # along independent of S/D alone (GQA16 vs GQA4 at fixed S/D).
    s = cfg["S"]
    d = cfg["D"]
    gqa = cfg.get("GQA", 1) or 1
    ws = s * d * (cfg.get("Hq", gqa) // gqa) * 2 * 2
    return [math.log2(s), math.log2(d), math.log2(cfg.get("P", 16)),
            cfg.get("B", 1), gqa, round(math.log2(max(ws, 1)), 3),
            round(math.log2(max(gqa * d, 1)), 3)]


def features11(cfg: Dict[str, Any], facts: Optional[Dict] = None) -> list:
    """V21-D (0a dual-source, schema v8): the 7 static geometry dims
    plus the four CUPTI runtime-truth permilles (stall/l2/warp/sm) —
    the USER-DEFINED core innovation: the mapping must distinguish on
    runtime truth, not geometry alone.

    Runtime dims missing (no trigger-window facts) -> returns the plain
    7-dim vector: the v8 kNN pads the missing columns with prototype
    means (a neutral centre) and counts it — the transient safety net;
    the acceptance evidence (observation non-empty rate) lives in
    drift_stats, and a steady-state miss means the COLLECTION chain is
    broken (fix the chain, never widen the fallback)."""
    x = features(cfg)
    if not isinstance(facts, dict):
        return x
    vals = [facts.get(k) for k in _RUNTIME_KEYS]
    if not all(isinstance(v, (int, float)) for v in vals):
        return x
    return x + [float(v) for v in vals]


class LearnedPolicy:
    # path -> ((mtime_ns, size), LearnedPolicy) -- see load()
    _cache: Dict[str, tuple] = {}

    def __init__(self, data: Dict):
        self.kind = data.get("kind")
        self.variants = data.get("variants") or {}
        self.tree = data.get("rules") or []
        self.prototypes = data.get("prototypes") or []
        self.k = int(data.get("k", 3))
        self.max_dist = data.get("max_dist")
        self.default_pick = data.get("default", "theory")
        # V17 S1-3 ①: explicit feature-schema versioning replaces the
        # silent zip() truncation of _predict_knn.  4 = pre-T6 models,
        # 6 = V16-T6 (adds GQA + log2 KV working-set), 7 = V18 B3 v7
        # (adds the GQA x D interaction log2(GQA*D)), 8 = V21-D
        # (adds the four CUPTI runtime permilles — dual-source).  A
        # missing field is INFERRED from the prototype width and
        # counted; with PACT_STRICT_SCHEMA=1 a mismatch refuses to
        # load instead.
        self.feature_schema_version = data.get("feature_schema_version")
        if self.feature_schema_version is None and self.prototypes:
            w = len(self.prototypes[0].get("feat") or [])
            self.feature_schema_version = w
        self.schema_padded = 0   # count of padded predictions (visibility)

    @classmethod
    def load(cls, path: Optional[str] = None) -> Optional["LearnedPolicy"]:
        path = path or os.environ.get("PACT_FAMILY_MODEL")
        if not path or not Path(path).is_file():
            return None
        # V16-T2b: the aobo decide path called this every forward, paying
        # a file read + json parse each time; cache on (mtime, size) so a
        # swapped model file is still picked up immediately
        try:
            st = os.stat(path)
            key = (path, st.st_mtime_ns, st.st_size)
        except OSError:
            key = None
        cached = cls._cache.get(path) if key else None
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            obj = cls(json.loads(Path(path).read_text()))
            # V17 S1-3 ①: strict mode refuses schema-mismatched models
            # outright instead of padding at predict time
            if os.environ.get("PACT_STRICT_SCHEMA") == "1" and \
                    obj.feature_schema_version not in (None, 4, 6, 7, 8):
                _SWALLOWED["schema_refused"] += 1
                return None
            if key is not None:
                cls._cache[path] = (key, obj)
            return obj
        except Exception as e:  # noqa: BLE001 - counted, never silent
            _SWALLOWED["load_errors"] += 1
            _SWALLOWED["last_error"] = str(e)[:120]
            return None

    def _predict_tree(self, x):
        node = self.tree[0] if self.tree else None
        # rules: nested via lo/hi indices into the list
        while node is not None and "pick" not in node:
            go_lo = x[node["feat"]] <= node["thr"]
            nxt = node.get("lo" if go_lo else "hi")
            if nxt is None:
                return "vanilla", 1.0
            node = self.tree[nxt] if isinstance(nxt, int) else None
        return (node or {}).get("pick", "vanilla"), 1.0

    def _predict_knn(self, x):
        import statistics
        # V17 S1-3 ①: EXPLICIT pad instead of zip() truncation.  A model
        # with fewer features than the runtime vector pads its prototype
        # columns with the per-column MEAN over prototypes (a neutral
        # centre for the L1 distance), and every padded prediction is
        # counted; V21-D: a model with MORE features (schema v8) than
        # the runtime vector (no trigger-window facts) pads the RUNTIME
        # side the same way — the static-7 degradation keeps using the
        # static columns instead of jumping to the default pick.  Silent
        # dimension dropping is gone.
        want = len(x)
        feats = [p["feat"] for p in self.prototypes]
        width = len(feats[0]) if feats else want
        if width < want:
            cols = list(zip(*feats))
            means = [sum(c) / len(c) for c in cols]
            feats = [list(f) + means[width:] for f in feats]
            self.schema_padded += 1
        elif width > want:
            cols = list(zip(*feats))
            x = list(x) + [sum(c) / len(c) for c in cols[want:]]
            self.schema_padded += 1
        ds = []
        for p, f in zip(self.prototypes, feats):
            d = sum(abs(a - b) for a, b in zip(x, f))
            ds.append((d, p.get("gains", {})))
        ds.sort(key=lambda t: t[0])
        # V9: outside the offline-measured neighbourhood, fall back to the
        # model's safe default (the static floor) instead of extrapolating.
        max_dist = getattr(self, "max_dist", None)
        default = getattr(self, "default_pick", "theory")
        if max_dist is not None and ds and ds[0][0] > max_dist:
            return default, 1.0
        top = ds[: self.k]
        # V19 B-4 (PACT_KNN_SOFTBAND=1, default OFF): distance prior +
        # default fallback band.  The neighbour median becomes
        # distance-weighted (1/(1+d)); an arm whose weighted gain does
        # not clear the 1.05 gate no longer wins -- an unconfident
        # extrapolation never risks a regression (the V18 holdout 3/24
        # shape).  Env off = the frozen median path, bit-for-bit.
        soft = os.environ.get("PACT_KNN_SOFTBAND") == "1"
        best, best_v = default, 1.0
        for v in top[0][1]:
            if soft:
                wsum = wtot = 0.0
                for d, g in top:
                    if v in g:
                        w = 1.0 / (1.0 + d)
                        wsum += w * g[v]
                        wtot += w
                if not wtot:
                    continue
                m = wsum / wtot
                if m <= 1.05:
                    continue          # below the gate: keep the safe default
            else:
                vals = [g.get(v, 1.0) for _, g in top if v in g]
                if not vals:
                    continue
                m = statistics.median(vals)
            if m > best_v:
                best, best_v = v, m
        return best, best_v

    def predict(self, cfg: Dict[str, Any],
                facts: Optional[Dict] = None) -> Tuple[str, float]:
        # V21-D: the dual-source vector — facts (CUPTI permilles) extend
        # the static geometry whenever the trigger window provided them
        x = features11(cfg, facts)
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
           cfg: Optional[Dict[str, Any]] = None,
           kv_heads: Optional[int] = None) -> Dict[str, Any]:
    """Same contract as online_decider.decide, with the learned policy first.

    The caller (service_main) may pass the kernel geometry ``cfg``
    ({D,P,S,B,Hq,GQA}); without it the learned policy is skipped and the
    FamilyTable path behaves exactly as before.

    V10-P0 (BEH-2): when ``cfg`` carries D/GQA they also address the
    ``d{D}g{gqa}|``-prefixed FamilyTable entries on the table fallback path
    (explicit head_dim/gqa arguments still win).  Previously the service
    path looked the plain keys up only, so prefixed entries were
    unaddressable there (the v7 dead-key lesson in residual form).

    V14-B: kv_heads (explicit or derived Hq//GQA from cfg) reaches the
    hints so the S1 residency working set covers all KV heads.
    """
    from triton.pact.decider.online_decider import decide as table_decide
    if cfg is None:
        return table_decide(facts, batch, seq_len, table=table,
                            hints_dir=hints_dir, head_dim=head_dim, gqa=gqa,
                            kv_heads=kv_heads)
    if head_dim is None and cfg.get("D"):
        head_dim = int(cfg["D"])
    if gqa is None and cfg.get("GQA"):
        gqa = int(cfg["GQA"])
    if kv_heads is None and cfg.get("Hq") and cfg.get("GQA"):
        try:
            kv_heads = int(cfg["Hq"]) // int(cfg["GQA"])
        except (TypeError, ValueError, ZeroDivisionError):
            kv_heads = None
    policy = LearnedPolicy.load()
    if policy is not None:
        try:
            # V19 N8 (PACT_DECIDER=exp3): the per-bucket bandit picks the
            # arm INSTEAD of the k-NN prediction (parallel selection, not
            # a blend -- the bandit's reward stream is what teaches it);
            # env resolution + the fallback contract stay identical
            variant = None
            if os.environ.get("PACT_DECIDER") == "exp3":
                bandit = _exp3_for(cfg, policy)
                variant = bandit.select()
            if variant is None:
                variant, _ = policy.predict(cfg, facts)
            env = policy.env_for(variant)
            if env is not None:
                if variant == "vanilla":
                    return {"family": "vanilla",
                            "extra_env": {"PACT_ENABLE": "0"},
                            "options_override": {}, "hints": {}}
                out = table_decide(facts, batch, seq_len, table=table,
                                   hints_dir=hints_dir, head_dim=head_dim,
                                   gqa=gqa, kv_heads=kv_heads)
                out["family"] = variant
                out["extra_env"] = dict(env)
                return out
        except Exception as e:  # noqa: BLE001 - counted, never silent
            _SWALLOWED["predict_errors"] += 1
            _SWALLOWED["last_error"] = str(e)[:120]
    return table_decide(facts, batch, seq_len, table=table, hints_dir=hints_dir,
                        head_dim=head_dim, gqa=gqa, kv_heads=kv_heads)


# ---------------------------------------------------------------------------
# V19 N8 (PACT_DECIDER=exp3, default OFF): EXP3 online arm selection
# (BanditSpec, ICML'25) running PARALLEL to the k-NN table.  One bandit
# per (B,S) bucket (non-stationary loads explore within a bucket, never
# across); arms = the loaded model's variant set + vanilla.  Reward is
# the interleaved paired difference from the G4 re-measurement, fed by
# the service layer (feed_exp3_reward); an SLO-window observation passes
# reward=None and updates NOTHING (the epsilon-clamp of the plan).
# ---------------------------------------------------------------------------
class Exp3Bandit:
    def __init__(self, arms, gamma=0.1, seed=0):
        import random
        self.arms = list(arms) or ["vanilla"]
        if "vanilla" not in self.arms:
            self.arms.append("vanilla")
        self.gamma = float(gamma)
        self.w = {a: 1.0 for a in self.arms}
        self.rng = random.Random(seed)
        self.picks = {a: 0 for a in self.arms}
        self.updates = 0

    def probs(self):
        tot = sum(self.w.values())
        k = len(self.arms)
        g = self.gamma
        return {a: (1.0 - g) * self.w[a] / tot + g / k
                for a in self.arms}

    def select(self):
        p = self.probs()
        x = self.rng.random()
        acc = 0.0
        for a in self.arms:
            acc += p[a]
            if x <= acc:
                self.picks[a] += 1
                return a
        a = self.arms[-1]
        self.picks[a] += 1
        return a

    def update(self, arm, reward):
        """reward = paired ratio (1.0 = parity, >1 = arm faster).
        None (SLO window / no measurement) updates nothing."""
        if reward is None or arm not in self.w:
            return
        r = min(max((reward - 1.0) / 2.0 + 0.5, 0.0), 1.0)  # -> [0,1]
        p = self.probs().get(arm) or 1.0
        k = len(self.arms)
        self.w[arm] *= __import__("math").exp(
            self.gamma * (r / p) / k)
        # drift clamp: no arm weight above 10x the total (EXP3 practice)
        tot = sum(self.w.values())
        cap = 10.0 * tot / k
        self.w = {a: min(v, cap) for a, v in self.w.items()}
        self.updates += 1


_EXP3: dict = {}


def _exp3_for(cfg: Dict[str, Any], policy) -> Optional[Exp3Bandit]:
    from triton.pact.runtime.workload_sniffer import bucket_bs
    key = bucket_bs(int(cfg.get("B") or 1), int(cfg.get("S") or 0))
    b = _EXP3.get(key)
    if b is None:
        arms = [a for a in (policy.variants or {}) if a != "vanilla"]
        # V21-D (F4): deterministic seed — str hash() is salted by
        # PYTHONHASHSEED, so cross-process selection sequences were not
        # reproducible (against the bit-for-bit discipline).  sha256 of
        # the bucket key is stable everywhere.
        import hashlib
        seed = int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16)
        b = Exp3Bandit(arms, seed=seed & 0xFFFF)
        _EXP3[key] = b
    return b


def feed_exp3_reward(cfg: Dict[str, Any], arm: str, reward) -> bool:
    """Service-layer hook: report the G4 paired ratio for the arm EXP3
    installed in this bucket.  No bandit yet / unknown arm -> False."""
    from triton.pact.runtime.workload_sniffer import bucket_bs
    key = bucket_bs(int(cfg.get("B") or 1), int(cfg.get("S") or 0))
    b = _EXP3.get(key)
    if b is None:
        return False
    b.update(arm, reward)
    return True


def exp3_state() -> Dict[str, Any]:
    """Observability snapshot (never a decision input elsewhere)."""
    return {k: {"picks": b.picks, "updates": b.updates,
                "probs": {a: round(p, 4) for a, p in b.probs().items()}}
            for k, b in _EXP3.items()}
