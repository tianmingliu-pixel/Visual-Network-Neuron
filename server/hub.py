"""
消息中心：线程安全地把训练线程的数据推给所有浏览器连接（SSE）
Hub: thread-safe fan-out from the training thread to every SSE client, with throttling.
"""
from __future__ import annotations

import asyncio
import json
import math
import threading
from typing import List, Optional, Set

FLUSH_INTERVAL = 0.1        # 指标合并周期（秒）→ 最多 10 条/秒 / metrics flush period
MAX_POINTS_PER_FLUSH = 60   # 每次最多发送的点数（超出则均匀抽稀）/ decimation cap
QUEUE_SIZE = 400            # 每个客户端的队列长度；慢客户端会丢弃旧指标而不是拖慢训练


def safe(obj):
    """递归替换 NaN/Inf 为 None，保证 JSON 合法 / make JSON-safe."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe(v) for v in obj]
    return obj


def encode(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(safe(data), ensure_ascii=False, allow_nan=False)}\n\n"


class Hub:
    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.subs: Set[asyncio.Queue] = set()
        self._metrics: List[dict] = []
        self._lock = threading.Lock()
        self.last = {}                      # 每类事件的最新一条，新连接时补发 / replay latest on connect

    def attach(self, loop):
        self.loop = loop

    # ---- 训练线程调用 / called from the training thread ------------------------------
    def publish(self, event: str, data):
        if event in ("status", "preview", "trace", "reset", "graph"):
            self.last[event] = data
            if event == "reset":
                for k in ("trace", "preview", "graph"):
                    self.last.pop(k, None)
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._fanout, encode(event, data), event == "metrics")

    def add_metric(self, point: dict):
        with self._lock:
            self._metrics.append(point)

    # ---- 事件循环内 / inside the event loop ---------------------------------------------
    def _fanout(self, msg: str, droppable: bool):
        for q in list(self.subs):
            if q.full():
                if droppable:
                    continue                 # 丢弃这批指标 / drop metrics for slow clients
                try:
                    q.get_nowait()           # 腾出位置给重要消息 / make room for important events
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(msg)

    async def flusher(self):
        while True:
            await asyncio.sleep(FLUSH_INTERVAL)
            with self._lock:
                pts, self._metrics = self._metrics, []
            if not pts:
                continue
            if len(pts) > MAX_POINTS_PER_FLUSH:          # 均匀抽稀，保留最后一个点
                k = len(pts) / MAX_POINTS_PER_FLUSH
                pts = [pts[int(i * k)] for i in range(MAX_POINTS_PER_FLUSH - 1)] + [pts[-1]]
            self._fanout(encode("metrics", {"points": pts}), True)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        for ev in ("status", "preview", "graph", "trace"):
            if ev in self.last:
                q.put_nowait(encode(ev, self.last[ev]))
        self.subs.add(q)
        return q

    def unsubscribe(self, q):
        self.subs.discard(q)
