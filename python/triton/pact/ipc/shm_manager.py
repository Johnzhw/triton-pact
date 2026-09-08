"""POSIX shared-memory blobs: JSON header + cubin bytes.

Layout: 8-byte big-endian header_len, UTF-8 JSON header, then raw cubin.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, Optional


@dataclass
class ShmBlob:
    shm_name: str
    shm_size: int
    header: Dict[str, Any]
    payload: bytes


def pack_blob(header: Dict[str, Any], payload: bytes) -> bytes:
    raw_header = json.dumps(header, sort_keys=True).encode("utf-8")
    return len(raw_header).to_bytes(8, "big") + raw_header + payload


def unpack_blob(blob: bytes) -> tuple[Dict[str, Any], bytes]:
    if len(blob) < 8:
        raise ValueError("blob too short")
    n = int.from_bytes(blob[:8], "big")
    if n < 0 or 8 + n > len(blob):
        raise ValueError("invalid header length")
    header = json.loads(blob[8:8 + n].decode("utf-8"))
    return header, blob[8 + n:]


class ShmManager:
    """Create / map / refcount-release shared memory segments.

    The creating process (compiler service) owns the SharedMemory objects
    until every consumer sends release_shm (or close() is called).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._owned: Dict[str, shared_memory.SharedMemory] = {}
        self._refs: Dict[str, int] = {}

    def create(self, name: str, header: Dict[str, Any], payload: bytes) -> ShmBlob:
        packed = pack_blob(header, payload)
        shm = shared_memory.SharedMemory(name=name, create=True, size=len(packed))
        shm.buf[:len(packed)] = packed
        with self._lock:
            self._owned[name] = shm
            self._refs[name] = 1
        return ShmBlob(shm_name=name, shm_size=len(packed), header=header,
                       payload=payload)

    def open(self, name: str) -> ShmBlob:
        shm = shared_memory.SharedMemory(name=name, create=False)
        try:
            packed = bytes(shm.buf)
            header, payload = unpack_blob(packed)
            with self._lock:
                self._refs[name] = self._refs.get(name, 0) + 1
            return ShmBlob(shm_name=name, shm_size=len(packed), header=header,
                           payload=payload)
        finally:
            shm.close()

    def add_ref(self, name: str) -> None:
        with self._lock:
            if name not in self._owned and name not in self._refs:
                raise KeyError(name)
            self._refs[name] = self._refs.get(name, 0) + 1

    def release(self, name: str) -> bool:
        """Decrement refcount.  Unlink when it hits 0.  Returns True if unlinked."""
        with self._lock:
            n = self._refs.get(name, 0) - 1
            if n > 0:
                self._refs[name] = n
                return False
            self._refs.pop(name, None)
            shm = self._owned.pop(name, None)
        if shm is not None:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        else:
            try:
                shared_memory.SharedMemory(name=name).unlink()
            except FileNotFoundError:
                pass
        return True

    def close(self) -> None:
        with self._lock:
            names = list(self._owned.keys())
        for name in names:
            self._refs[name] = 1
            self.release(name)
