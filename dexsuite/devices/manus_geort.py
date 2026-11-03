"""Manus glove streaming client (WebSocket).

This module provides a small helper to receive joint positions streamed over a
WebSocket connection (typically from an external Manus service) and expose them
as a numpy array.

The websockets dependency is optional; it is only required if you actually
instantiate ManusGeort.
"""

from __future__ import annotations

import asyncio
import json
import threading

import numpy as np

SAPIEN_TO_DEX = np.array(
    [
        0,
        4,
        8,
        12,  # ffj0, mfj0, rfj0, thj0
        1,
        5,
        9,
        13,  # ffj1, mfj1, rfj1, thj1
        2,
        6,
        10,
        14,  # ffj2, mfj2, rfj2, thj2
        3,
        7,
        11,
        15,  # ffj3, mfj3, rfj3, thj3
    ],
    dtype=int,
)


class ManusGeort:
    def __init__(
        self,
        url: str,
        expected_dim: int,
        reconnect_sec: float = 0.5,
        to_dex: np.array = SAPIEN_TO_DEX,
    ):
        try:
            import websockets  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "ManusGeort requires 'websockets' (pip install websockets).",
            ) from e

        self.url = url
        self.expected_dim = int(expected_dim)
        self.reconnect_sec = float(reconnect_sec)
        self.latest = None
        self._stop = False
        self._loop = asyncio.new_event_loop()
        self._thr = threading.Thread(target=self._run_loop, daemon=True)
        self._thr.start()
        self.to_dex = to_dex
        self.remap = len(to_dex) == expected_dim

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._consumer())

    async def _consumer(self):
        import websockets

        while not self._stop:
            try:
                async with websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=20,
                    max_queue=1,
                ) as ws:
                    # optional: read and ignore meta/hello
                    async for msg in ws:
                        try:
                            obj = json.loads(msg)
                            if obj.get("type") == "qpos":
                                q = obj.get("qpos", [])
                                if isinstance(q, list):
                                    self.latest = q
                        except Exception:
                            pass
            except Exception:
                await asyncio.sleep(self.reconnect_sec)

    def get(self) -> np.ndarray:
        if self.latest is None:
            return np.zeros(self.expected_dim, dtype=np.float32)
        q = np.asarray(self.latest, dtype=np.float32).reshape(-1)
        if q.size < self.expected_dim:
            q = np.pad(q, (0, self.expected_dim - q.size))
        elif q.size > self.expected_dim:
            q = q[: self.expected_dim]
        return q

    def close(self):
        self._stop = True
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass

    def get_action(self):
        q_sap = self.get()  # list/ndarray, SAPIEN order
        if self.remap:
            q_dex = self._remap_sapien_to_dex(q_sap)  # (16,)
            return q_dex
        return q_sap

    def _remap_sapien_to_dex(self, q_sap: np.ndarray) -> np.ndarray:
        q_sap = np.asarray(q_sap, dtype=np.float32).reshape(-1)
        if q_sap.size < len(self.to_dex):
            # if the stream hiccups, pad to 16 first
            q_sap = np.pad(q_sap, (0, len(self.to_dex) - q_sap.size))
        q_sap = q_sap[: len(self.to_dex)]
        q_dex = q_sap[self.to_dex]
        return q_dex
