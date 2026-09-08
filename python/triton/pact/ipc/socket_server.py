"""Unix-domain JSON-line server for the compiler service process."""
from __future__ import annotations

import json
import os
import socket
import threading
from typing import Callable, Dict, Optional


Handler = Callable[[Dict], Dict]


class PactSocketServer:
    def __init__(self, path: str, handler: Handler):
        self.path = path
        self.handler = handler
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if os.path.exists(self.path):
            os.unlink(self.path)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(self.path)
        self._sock.listen(8)
        self._sock.settimeout(0.2)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="pact-ipc",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if os.path.exists(self.path):
            os.unlink(self.path)

    def _loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        try:
            with conn:
                buf = b""
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if not line.strip():
                            continue
                        req = json.loads(line.decode("utf-8"))
                        try:
                            resp = self.handler(req)
                        except Exception as e:
                            resp = {
                                "msg_id": req.get("msg_id"),
                                "type": "error",
                                "reason": str(e),
                            }
                        conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
        except OSError:
            pass
