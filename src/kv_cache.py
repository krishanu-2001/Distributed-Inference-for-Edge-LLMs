import uuid
from dataclasses import dataclass


@dataclass
class KVCacheEntry:
    cache_id: str
    token_count: int


class KVCacheManager:
    """Token-granular KV memory tracker.

    Each radix-tree node owns exactly one entry covering exactly its own edge
    tokens, so `used_tokens` always equals the actual tree's `total_tokens`.
    """

    def __init__(self, max_tokens: int):
        self._max_tokens = max_tokens
        self.used_tokens: int = 0
        self.entries: dict[str, KVCacheEntry] = {}

    def allocate(self, num_tokens: int) -> KVCacheEntry | None:
        """Allocate KV cache for num_tokens. Returns entry or None if no space."""
        if num_tokens <= 0 or self.used_tokens + num_tokens > self._max_tokens:
            return None

        cache_id = uuid.uuid4().hex[:16]
        entry = KVCacheEntry(cache_id=cache_id, token_count=num_tokens)
        self.entries[cache_id] = entry
        self.used_tokens += num_tokens
        return entry

    def split(self, cache_id: str, head_tokens: int) -> KVCacheEntry:
        """Split an entry in two: a new entry for the first `head_tokens` tokens,
        and the original entry shrunk to the remainder. Total usage is unchanged.
        Used when a radix-tree edge is split."""
        entry = self.entries[cache_id]
        if not 0 < head_tokens < entry.token_count:
            raise ValueError(
                f"cannot split {entry.token_count}-token entry at {head_tokens}"
            )
        head = KVCacheEntry(cache_id=uuid.uuid4().hex[:16], token_count=head_tokens)
        self.entries[head.cache_id] = head
        entry.token_count -= head_tokens
        return head

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
