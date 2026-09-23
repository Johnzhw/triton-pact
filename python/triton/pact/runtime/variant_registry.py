"""V16-T3: bounded variant bookkeeping with per-domain LRU eviction.

Why: the eager pools grow without bound today (ResidentPool._geo keeps
one kernel set per geometry; HotSwapper._pool is add-only), and the
user's point 4 asks for variant VERSIONING once the fixed vocabulary
caps come off (T3/T4 remove them).  This registry caps the LIVE set per
(model tag, geometry domain); evicted variants lose only their in-
process handles -- the triton disk cache keeps the cubin, so a re-decide
of an evicted variant revives in milliseconds via the normal compile
path (measured behaviour of the shared TRITON_CACHE_DIR hit).

Not a CUDA-side anything: pure bookkeeping, never on the launch path.
The pools consult it when they are about to add a NEW variant handle.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


class VariantRegistry:
    def __init__(self, max_per_domain: int = 32):
        self._lock = threading.Lock()
        self._max = int(max_per_domain) if max_per_domain else 32
        # domain -> OrderedDict[key -> meta]; move_to_end on hit (LRU)
        self._domains: Dict[Tuple[str, str], OrderedDict] = {}
        # evicted-but-not-yet-reaped reports: (domain, key, meta)
        self._evictions: List[Tuple[Tuple[str, str], str, Any]] = []
        self._generation = 0

    # -- keys ---------------------------------------------------------------
    @staticmethod
    def variant_key(name: str, extra_env: Dict[str, str],
                    options_override: Optional[Dict[str, Any]] = None) -> str:
        """Stable identity of a variant: name + normalised env/options.
        The compile cache key already folds env in (cache.py); this key
        mirrors that identity for the LIVE-set accounting."""
        import hashlib
        import json
        blob = json.dumps({
            "name": str(name),
            "env": {str(k): str(v) for k, v in sorted(
                (extra_env or {}).items())},
            "opts": {str(k): str(v) for k, v in sorted(
                (options_override or {}).items())},
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    # -- accounting ---------------------------------------------------------
    def register(self, domain: Tuple[str, str], key: str,
                 meta: Optional[Dict[str, Any]] = None
                 ) -> List[Tuple[str, Any]]:
        """Add (or refresh) a variant in a domain's live set.  Returns the
        (key, meta) pairs EVICTED from this domain (LRU order) -- callers
        reap the corresponding pool handles.  Re-registering an existing
        key is a hit (refreshes recency), matching pool-side
        resurrection."""
        evicted: List[Tuple[str, Any]] = []
        with self._lock:
            book = self._domains.setdefault(domain, OrderedDict())
            if key in book:
                book.move_to_end(key)
                if meta is not None:
                    book[key] = meta
                return evicted
            book[key] = meta or {}
            self._generation += 1
            while len(book) > self._max:
                old_key, old_meta = book.popitem(last=False)
                self._evictions.append((domain, old_key, old_meta))
                evicted.append((old_key, old_meta))
        return evicted

    def hit(self, domain: Tuple[str, str], key: str) -> bool:
        with self._lock:
            book = self._domains.get(domain)
            if book is None or key not in book:
                return False
            book.move_to_end(key)
            return True

    def contains(self, domain: Tuple[str, str], key: str) -> bool:
        with self._lock:
            return key in self._domains.get(domain, ())

    def drop_domain(self, domain: Tuple[str, str]) -> int:
        """Version invalidation: a model/config change retires the whole
        domain (cache keys already isolate binaries; this frees handles)."""
        with self._lock:
            return len(self._domains.pop(domain, ()))

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "generation": self._generation,
                "domains": {f"{d[0]}|{d[1]}": len(v)
                            for d, v in self._domains.items()},
                "total_evictions": len(self._evictions),
                "max_per_domain": self._max,
            }

    def drain_evictions(self) -> List[Tuple[Tuple[str, str], str, Any]]:
        with self._lock:
            out, self._evictions = self._evictions, []
            return out


# module-level default (process-wide); env-tunable for experiments
import os  # noqa: E402


def default_registry() -> VariantRegistry:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = VariantRegistry(
            max_per_domain=int(os.environ.get("PACT_VARIANT_CAP") or 32))
    return _DEFAULT


_DEFAULT: Optional[VariantRegistry] = None
