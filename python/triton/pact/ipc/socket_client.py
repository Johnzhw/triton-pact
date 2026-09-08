"""Non-blocking-friendly Unix-domain JSON-line client."""
from __future__ import annotations

import json
import socket
from typing import Dict, Optional


class PactSocketClient:
    def __init__(self, path: str, timeout: float = 0.05):
        self.path = path
        self.timeout = timeout

    def request(self, msg: Dict, timeout: Optional[float] = None) -> Dict:
        """Blocking request.  Raises on connection failure."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout if timeout is None else timeout)
        try:
            sock.connect(self.path)
            sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError("server closed")
                buf += chunk
            line = buf.split(b"\n", 1)[0]
            return json.loads(line.decode("utf-8"))
        finally:
            sock.close()

    def try_request(self, msg: Dict, timeout: Optional[float] = None) -> Optional[Dict]:
        """Non-blocking for the inference loop: None if the server is down."""
        try:
            return self.request(msg, timeout=timeout)
        except (OSError, ConnectionError, TimeoutError, json.JSONDecodeError):
            return None
