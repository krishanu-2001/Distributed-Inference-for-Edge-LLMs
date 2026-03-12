import uuid
from dataclasses import dataclass


@dataclass
class KVCacheEntry:
    cache_id: str
    token_count: int


class KVCacheManager:
    def __init__(self, max_tokens: int):
        self._max_tokens = max_tokens
        self.used_tokens: int = 0
        self.entries: dict[str, KVCacheEntry] = {}

    def allocate(self, num_tokens: int) -> KVCacheEntry | None:
        """Allocate KV cache for num_tokens. Returns entry or None if no space."""
        if self.used_tokens + num_tokens > self._max_tokens:
            return None

        cache_id = uuid.uuid4().hex[:16]
        entry = KVCacheEntry(
            cache_id=cache_id,
            token_count=num_tokens,
        )
        self.entries[cache_id] = entry
        self.used_tokens += num_tokens
        return entry

    def free(self, cache_id: str) -> int:
        """Free a cache entry. Returns tokens freed."""
        if cache_id not in self.entries:
            return 0
        entry = self.entries.pop(cache_id)
        self.used_tokens -= entry.token_count
        return entry.token_count

    @property
    def utilization(self) -> float:
        if self._max_tokens == 0:
            return 0.0
        return self.used_tokens / self._max_tokens

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def free_tokens(self) -> int:
        return self._max_tokens - self.used_tokens

    def stats(self) -> dict:
        return {
            "max_tokens": self._max_tokens,
            "used_tokens": self.used_tokens,
            "free_tokens": self.free_tokens,
            "utilization": round(self.utilization, 4),
            "num_entries": len(self.entries),
        }
