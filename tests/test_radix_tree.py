import random

import pytest

from src.kv_cache import KVCacheManager
from src.radix_tree import RadixTree


def make_tree(capacity=1000):
    kv = KVCacheManager(capacity)
    return RadixTree(kv_cache=kv), kv


def assert_consistent(tree: RadixTree, kv: KVCacheManager):
    """Tree token count == KV usage, and every node's KV entry matches its edge."""
    assert tree.total_tokens == kv.used_tokens
    seen = set()
    stack = list(tree.root.children.values())
    while stack:
        node = stack.pop()
        assert node.kv_cache_id in kv.entries
        assert kv.entries[node.kv_cache_id].token_count == len(node.token_ids)
        seen.add(node.kv_cache_id)
        stack.extend(node.children.values())
    assert seen == set(kv.entries)


def test_match_prefix_full_and_partial():
    tree, _ = make_tree()
    tree.insert([1, 2, 3, 4])
    assert tree.match_prefix([1, 2, 3, 4, 5])[0] == 4
    assert tree.match_prefix([1, 2, 9])[0] == 2  # partial edge counts
    assert tree.match_prefix([9])[0] == 0


def test_insert_returns_tokens_added_and_splits_kv():
    tree, kv = make_tree()
    assert tree.insert([1, 2, 3, 4]) == 4
    assert tree.insert([1, 2, 5, 6]) == 2  # split [1,2] | [3,4], add [5,6]
    assert tree.insert([1, 2]) == 0
    assert tree.total_tokens == 6
    assert_consistent(tree, kv)


def test_no_kv_leak_after_evicting_everything():
    # Previously leaked: merge on eviction dropped the parent's cache id.
    tree, kv = make_tree()
    for seq in ([1, 2, 3, 4], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 7, 8]):
        tree.insert(seq)
    while tree.evict_lru():
        assert_consistent(tree, kv)
    assert tree.total_tokens == 0
    assert kv.used_tokens == 0 and not kv.entries


def test_evict_lru_returns_zero_when_nothing_evictable():
    tree, _ = make_tree()
    assert tree.evict_lru() == 0
    tree.insert([1, 2, 3])
    _, nodes = tree.match_prefix([1, 2, 3])
    tree.inc_ref(nodes)
    assert tree.evict_lru() == 0
    tree.dec_ref(nodes)
    assert tree.evict_lru() == 3


def test_pinned_node_survives_split_and_eviction():
    tree, kv = make_tree()
    tree.insert([1, 2, 3, 4])
    _, pinned = tree.match_prefix([1, 2, 3, 4])
    tree.inc_ref(pinned)
    tree.insert([1, 2, 9, 9])  # splits the pinned edge
    while tree.evict_lru():
        pass
    # Only the pinned path ([1,2] + [3,4]) remains
    assert tree.match_prefix([1, 2, 3, 4])[0] == 4
    assert tree.total_tokens == 4
    tree.dec_ref(pinned)
    while tree.evict_lru():
        pass
    assert tree.total_tokens == 0 and kv.used_tokens == 0
    assert_consistent(tree, kv)


def test_lru_order_and_touch_flag():
    tree, _ = make_tree()
    tree.insert([1, 1])
    tree.insert([2, 2])
    tree.root.children[1].last_access = 0
    tree.root.children[2].last_access = 1
    tree.match_prefix([1, 1], touch=False)  # read-only probe: no LRU update
    assert tree.evict_lru() == 2
    assert 1 not in tree.root.children


def test_insert_raises_when_kv_full():
    tree, _ = make_tree(capacity=3)
    with pytest.raises(MemoryError):
        tree.insert([1, 2, 3, 4])


def test_random_workload_stays_consistent():
    rng = random.Random(0)
    tree, kv = make_tree(capacity=200)
    prefixes = [[rng.randrange(20) for _ in range(rng.randrange(1, 15))] for _ in range(8)]
    for _ in range(500):
        seq = rng.choice(prefixes) + [rng.randrange(20) for _ in range(rng.randrange(0, 10))]
        _, pinned = tree.match_prefix(seq)
        tree.inc_ref(pinned)
        needed = len(seq) - tree.match_prefix(seq, touch=False)[0]
        while kv.free_tokens < needed and tree.evict_lru():
            pass
        if kv.free_tokens >= needed:
            assert tree.insert(seq) == needed
        tree.dec_ref(pinned)
        assert_consistent(tree, kv)
        assert kv.used_tokens <= kv.max_tokens


def test_approx_tree_without_kv():
    tree = RadixTree()
    tree.insert([1, 2, 3])
    tree.insert([1, 2, 4])
    assert tree.total_tokens == 4
    assert sorted(tree.get_all_sequences()) == [[1, 2, 3], [1, 2, 4]]
    assert tree.evict_lru() == 1
