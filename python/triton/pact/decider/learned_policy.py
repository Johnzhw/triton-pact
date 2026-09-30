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

FEATURE_ORDER = ("logS", "logD", "logP", "B")


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
        # (adds the GQA x D interaction log2(GQA*D)).  A missing field
        # is INFERRED from the prototype width and counted; with
        # PACT_STRICT_SCHEMA=1 a mismatch refuses to load instead.
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
                    obj.feature_schema_version not in (None, 4, 6, 7):
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
        # V17 S1-3 ①: EXPLICIT pad instead of zip() truncation.  A model
        # with fewer features than the runtime vector pads its prototype
        # columns with the per-column MEAN over prototypes (a neutral
        # centre for the L1 distance), and every padded prediction is
        # counted; more features than runtime is a hard mismatch ->
        # default pick (counted).  Silent dimension dropping is gone.
        want = len(x)
        feats = [p["feat"] for p in self.prototypes]
        width = len(feats[0]) if feats else want
        if width < want:
            cols = list(zip(*feats))
            means = [sum(c) / len(c) for c in cols]
            feats = [list(f) + means[width:] for f in feats]
            self.schema_padded += 1
        elif width > want:
            self.schema_padded += 1
            return self.default_pick, 1.0
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
            variant, _ = policy.predict(cfg)
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
