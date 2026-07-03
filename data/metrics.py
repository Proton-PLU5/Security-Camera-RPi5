import time
import logging
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Deque, Dict

@dataclass
class MetricStats:
    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = 0.0

    # rolling window of recent samples
    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
 
    def add(self, value: float):
        self.count += 1
        self.total += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.samples.append(value)
 
    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0
 
    def recent_avg(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0
 
    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        idx = min(int(len(s) * p), len(s) - 1)
        return s[idx]

class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.stats: Dict[str, MetricStats] = {}
        self.counters: Dict[str, int] = {}

    def record(self, name: str, value: float):
        """Record one timing/value sample under `name`, in seconds."""
        with self.lock:
            self.stats.setdefault(name, MetricStats()).add(value)
 
    def increment(self, name: str, amount: int = 1):
        """Bump a simple event counter (e.g. dropped frames, reconnects)."""
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + amount
    
    @contextmanager
    def time(self, name: str):
        """Context manager to time a block of code and record it under `name`."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.record(name, elapsed)

    def snapshot(self) -> Dict[str, Dict[str, dict]]:
        """Return a snapshot of all metrics."""
        with self.lock:
            out = {}
            for name, stat in self.stats.items():
                out[name] = {
                    "count": stat.count,
                    "avg_ms": stat.avg * 1000,
                    "recent_avg_ms": stat.recent_avg() * 1000,
                    "p95_ms": stat.percentile(0.95) * 1000,
                    "min_ms": (stat.min if stat.min != float("inf") else 0) * 1000,
                    "max_ms": stat.max * 1000,
                }
            for name, count in self.counters.items():
                out[name] = {"count": count}
            return out
        
    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.stats.clear()
            self.counters.clear()

    def report(self):
        """Log a summary of all metrics."""
        snapshot = self.snapshot()
        for name, data in snapshot.items():
            if "avg_ms" in data:
                logging.info(
                    f"{name}: count={data['count']}, avg={data['avg_ms']:.2f}ms, "
                    f"recent_avg={data['recent_avg_ms']:.2f}ms, p95={data['p95_ms']:.2f}ms, "
                    f"min={data['min_ms']:.2f}ms, max={data['max_ms']:.2f}ms"
                )
            else:
                logging.info(f"{name}: count={data['count']}")

# Global metrics instance
metrics = Metrics()