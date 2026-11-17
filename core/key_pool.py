"""
core/key_pool.py
----------------
Multi-key pooling, rotation, and rate-limit cooldown manager for Groq API.
Supports single or multi-key configurations (comma, newline, or semicolon separated).
Tracks per-key requests, latencies, and cooldowns with zero-leak key masking.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Any

try:
    from groq import AsyncGroq, Groq
except ImportError:
    AsyncGroq = None  # type: ignore[assignment, misc]
    Groq = None  # type: ignore[assignment, misc]

from core.diagnostics import session_diagnostics


def mask_api_key(key: str) -> str:
    """Mask an API key for safe UI and logging display (e.g. gsk_...3a8f)."""
    if not key:
        return "empty"
    clean = key.strip().strip("'\"")
    if len(clean) <= 10:
        return clean
    prefix = clean[:4] if clean.startswith("gsk_") else clean[:3]
    suffix = clean[-4:]
    return f"{prefix}...{suffix}"


def parse_api_keys(raw_input: str | None) -> list[str]:
    """
    Parse one or multiple API keys from env strings or user input.
    Accepts comma, newline, or semicolon separated values.
    Filters out empty strings, comments, and invalid placeholders.
    """
    if not raw_input or not isinstance(raw_input, str):
        return []

    # Split on commas, semicolons, and newlines
    tokens = re.split(r"[,;\n\r]+", raw_input)
    keys: list[str] = []
    seen = set()

    for t in tokens:
        clean = t.strip().strip("'\"")
        if not clean or clean.lower().startswith("your_"):
            continue
        # Groq API keys start with gsk_ and are >= 30 chars
        if (clean.startswith("gsk_") or len(clean) >= 30) and len(clean) >= 25 and clean not in seen:
            if clean.startswith("gsk_") or len(clean) >= 30:
                keys.append(clean)
                seen.add(clean)

    return keys


@dataclass
class KeySlot:
    """Represents a single Groq API key slot in the pool with health and usage metrics."""

    index: int  # 1-based index
    key: str
    masked_key: str = field(init=False)
    _client_async: Optional[Any] = field(default=None, repr=False)
    _client_sync: Optional[Any] = field(default=None, repr=False)
    last_used_time: float = 0.0
    cooldown_until: float = 0.0
    requests_count: int = 0
    error_count: int = 0

    def __post_init__(self) -> None:
        self.masked_key = mask_api_key(self.key)

    @property
    def is_cooling(self) -> bool:
        """Returns True if this key is currently in a temporary rate-limit cooldown."""
        return time.time() < self.cooldown_until

    @property
    def cooldown_remaining(self) -> float:
        """Seconds remaining in cooldown, or 0.0 if active."""
        return max(0.0, self.cooldown_until - time.time())

    def get_async_client(self) -> Any:
        """Return the cached or newly initialized AsyncGroq client."""
        if self._client_async is None:
            if AsyncGroq is None:
                raise ImportError("The 'groq' package is not installed. Please run: pip install groq")
            self._client_async = AsyncGroq(api_key=self.key)
        return self._client_async

    def get_sync_client(self) -> Any:
        """Return the cached or newly initialized Groq client."""
        if self._client_sync is None:
            if Groq is None:
                raise ImportError("The 'groq' package is not installed. Please run: pip install groq")
            self._client_sync = Groq(api_key=self.key)
        return self._client_sync

    def record_success(self, latency_ms: float = 0.0) -> None:
        """Record a successful request completion."""
        self.requests_count += 1
        self.last_used_time = time.time()

    def record_error(self) -> None:
        """Record an error encounter."""
        self.error_count += 1
        self.last_used_time = time.time()


class GroqKeyPool:
    """Thread-safe & async-compatible pool for rotating across multiple Groq API keys."""

    def __init__(self, keys: list[str]) -> None:
        self._lock = threading.Lock()
        self.slots: list[KeySlot] = [
            KeySlot(index=i, key=k) for i, k in enumerate(keys, start=1)
        ]
        self._cursor: int = 0

    @property
    def size(self) -> int:
        return len(self.slots)

    @property
    def active_slots(self) -> list[KeySlot]:
        """Returns slots that are not currently cooling."""
        return [s for s in self.slots if not s.is_cooling]

    def acquire_key(self, wait_if_cooling: bool = False) -> KeySlot:
        """
        Acquire the next available key slot in round-robin sequence.
        If all keys are cooling and wait_if_cooling is True, sleeps until the earliest key recovers.
        """
        with self._lock:
            if not self.slots:
                raise ValueError("No Groq API keys configured in the pool.")

            n = len(self.slots)
            # Try to find a healthy (non-cooling) key starting from cursor
            for offset in range(n):
                idx = (self._cursor + offset) % n
                candidate = self.slots[idx]
                if not candidate.is_cooling:
                    self._cursor = (idx + 1) % n
                    return candidate

            # If all are cooling:
            # Find the slot with earliest cooldown expiration
            earliest_slot = min(self.slots, key=lambda s: s.cooldown_remaining)
            rem = earliest_slot.cooldown_remaining

        if wait_if_cooling and rem > 0:
            session_diagnostics.warning(
                "KEY-POOL",
                f"All {len(self.slots)} Groq keys cooling. Waiting {rem:.1f}s for Key {earliest_slot.index}...",
            )
            time.sleep(rem + 0.1)
            return earliest_slot

        # If not waiting, return the earliest cooling slot anyway as best-effort fallback
        return earliest_slot

    def mark_rate_limited(self, index: int, cooldown_seconds: float = 60.0) -> None:
        """Put a key into temporary quarantine upon encountering HTTP 429."""
        with self._lock:
            for s in self.slots:
                if s.index == index:
                    s.cooldown_until = time.time() + cooldown_seconds
                    s.record_error()
                    session_diagnostics.warning(
                        "KEY-POOL",
                        f"Key {s.index} ({s.masked_key}) rate-limited (HTTP 429). Cooling for {cooldown_seconds:.0f}s. "
                        f"Auto-rerouting requests to other active keys...",
                        details={"key_index": s.index, "cooldown_sec": cooldown_seconds},
                    )
                    break

    def get_status_summary(self) -> list[dict]:
        """Return a snapshot of all keys and their live metrics for the UI."""
        summary = []
        for s in self.slots:
            cooling = s.is_cooling
            rem = s.cooldown_remaining
            status_str = f"Cooling ({rem:.0f}s left)" if cooling else "Active"
            summary.append(
                {
                    "index": s.index,
                    "masked_key": s.masked_key,
                    "status": status_str,
                    "is_cooling": cooling,
                    "requests_count": s.requests_count,
                    "error_count": s.error_count,
                }
            )
        return summary
