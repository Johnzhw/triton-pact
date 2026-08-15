"""PACT PGO profile database.

Persists derived hardware facts keyed by (kernel source hash, constexpr
signature, shape configuration) so a profile collected for one decode shape
can be reused by later runs of the same kernel.
"""
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional


def default_db_dir() -> Path:
    base = os.environ.get("PACT_PGO_DB_DIR")
    if base:
        return Path(base)
    cache = os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton" / "cache"))
    return Path(cache) / "pact_pgo"


class PactProfileDB:
    def __init__(self, db_dir: Optional[Path] = None):
        self.db_dir = Path(db_dir or default_db_dir())
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.db_dir / "profiles.json"
        self._lock = threading.Lock()

    def _read(self) -> Dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            with open(self._path) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def get(self, key: str) -> Dict[str, Any]:
        with self._lock:
            data = self._read()
            return dict(data.get(key, {}))

    def put(self, key: str, facts: Dict[str, Any]) -> None:
        with self._lock:
            self.db_dir.mkdir(parents=True, exist_ok=True)
            data = self._read()
            entry = dict(data.get(key, {}))
            entry.update(facts)
            entry["updated_at"] = __import__("time").time()
            data[key] = entry
            tmp = self._path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(data, f)
            tmp.replace(self._path)

    def keys(self):
        return list(self._read().keys())
