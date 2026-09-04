"""Lightweight machine telemetry for bounded Lightning RLRMB runs."""
from __future__ import annotations

import json
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

CpuTicks = tuple[tuple[int, int], ...]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_cpu_ticks(path: Path = Path("/proc/stat")) -> CpuTicks:
    rows: list[tuple[int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("cpu"):
            if rows:
                break
            continue
        if fields[0] != "cpu" and not fields[0][3:].isdigit():
            continue
        values = [int(value) for value in fields[1:]]
        if len(values) < 4:
            raise ValueError("/proc/stat CPU row is incomplete")
        total = sum(values)
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        rows.append((total, idle))
    if not rows:
        raise ValueError("/proc/stat contains no CPU rows")
    return tuple(rows)


def cpu_percentages(previous: Sequence[tuple[int, int]], current: Sequence[tuple[int, int]]) -> tuple[float, tuple[float, ...]]:
    """Return aggregate-machine and per-core busy percentages from tick deltas."""
    if len(previous) != len(current) or not current:
        raise ValueError("CPU tick snapshots must have the same non-zero shape")
    values: list[float] = []
    for prior, now in zip(previous, current):
        total_delta = int(now[0]) - int(prior[0])
        idle_delta = int(now[1]) - int(prior[1])
        if total_delta <= 0 or idle_delta < 0 or idle_delta > total_delta:
            values.append(0.0)
            continue
        values.append(100.0 * (total_delta - idle_delta) / total_delta)
    return values[0], tuple(values[1:])


def _read_ram(path: Path = Path("/proc/meminfo")) -> dict[str, float | int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        fields = raw.split()
        if fields:
            values[key] = int(fields[0]) * 1024
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None or total <= 0:
        raise ValueError("/proc/meminfo lacks MemTotal/MemAvailable")
    used = max(0, total - available)
    return {"used_bytes": used, "total_bytes": total, "percent": 100.0 * used / total}


def _optional_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped or stripped.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _read_gpu() -> dict[str, float | None]:
    command = (
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,utilization.memory,power.draw",
        "--format=csv,noheader,nounits",
    )
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=2.0)
    except (OSError, subprocess.SubprocessError):
        return {
            "utilization_percent": None,
            "vram_used_mib": None,
            "vram_total_mib": None,
            "memory_utilization_percent": None,
            "power_draw_watts": None,
        }
    first = next((line for line in result.stdout.splitlines() if line.strip()), "")
    fields = [part.strip() for part in first.split(",")]
    if len(fields) != 5:
        return {
            "utilization_percent": None,
            "vram_used_mib": None,
            "vram_total_mib": None,
            "memory_utilization_percent": None,
            "power_draw_watts": None,
        }
    values = [_optional_float(value) for value in fields]
    return {
        "utilization_percent": values[0],
        "vram_used_mib": values[1],
        "vram_total_mib": values[2],
        "memory_utilization_percent": values[3],
        "power_draw_watts": values[4],
    }


class _MachineSampleProvider:
    def __init__(self) -> None:
        self._previous_cpu: CpuTicks | None = None

    def __call__(self) -> Mapping[str, Any]:
        current = _read_cpu_ticks()
        previous = self._previous_cpu or current
        aggregate, cores = cpu_percentages(previous, current)
        self._previous_cpu = current
        return {
            "timestamp": _timestamp(),
            "gpu": _read_gpu(),
            "cpu": {"aggregate_percent": aggregate, "per_core_percent": list(cores)},
            "ram": _read_ram(),
        }


class TelemetrySampler:
    """Periodic compact JSONL sampler with deterministic start/stop lifecycle."""

    def __init__(self, path: Path, *, interval_seconds: float,
                 sample_provider: Callable[[], Mapping[str, Any]] | None = None) -> None:
        if interval_seconds <= 0:
            raise ValueError("telemetry interval must be positive")
        self.path = Path(path)
        self.interval_seconds = float(interval_seconds)
        self.sample_provider = sample_provider or _MachineSampleProvider()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None

    def start(self) -> "TelemetrySampler":
        if self._thread is not None:
            raise RuntimeError("telemetry sampler is already running")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rlrmb-telemetry", daemon=False)
        self._thread.start()
        return self

    def _run(self) -> None:
        assert self._stream is not None
        while not self._stop.is_set():
            try:
                document = dict(self.sample_provider())
                document.setdefault("timestamp", _timestamp())
            except Exception as exc:
                document = {"timestamp": _timestamp(), "telemetry_error": str(exc)}
            self._stream.write(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
            self._stream.flush()
            if self._stop.wait(self.interval_seconds):
                break

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        thread.join(timeout=3.0)
        if thread.is_alive():
            raise RuntimeError("telemetry sampler failed to terminate")
        self._thread = None
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "TelemetrySampler":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()
