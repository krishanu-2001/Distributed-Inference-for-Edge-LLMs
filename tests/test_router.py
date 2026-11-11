import subprocess
import sys

from src.radix_tree import RadixTree
from src.router import Router, mock_tokenize

NODES = [0, 1, 2, 3]


def make_router(own=1, policy="cache_aware"):
    local = RadixTree()
    return Router(own, NODES, policy=policy, local_tree=local), local


def test_uses_exact_local_tree_for_own_node():
    # Previously: own approx tree was always empty, so this forwarded to node 0.
    router, local = make_router(own=1)
    local.insert([1, 2, 3, 4, 5])
    router.update_approx_tree(0, [1, 2, 3])
    assert router.route([1, 2, 3, 4, 5]) == 1


def test_ties_stay_local():
    # Previously: max() picked the lowest node id on ties.
    router, local = make_router(own=2)
    local.insert([1, 2])
    router.update_approx_tree(0, [1, 2])
    assert router.route([1, 2, 9]) == 2


def test_no_match_anywhere_stays_local():
    router, _ = make_router(own=3)
    assert router.route([7, 8, 9]) == 3


def test_forwards_to_strictly_longer_match():
    router, local = make_router(own=0)
    local.insert([1, 2])
    router.update_approx_tree(2, [1, 2, 3])
    router.update_approx_tree(3, [1, 2, 3, 4])
    assert router.route([1, 2, 3, 4, 5]) == 3


def test_routing_does_not_touch_local_lru():
    router, local = make_router(own=1)
    local.insert([1, 2, 3])
    local.root.children[1].last_access = 0
    router.route([1, 2, 3])
    assert local.root.children[1].last_access == 0


def test_own_estimate_updates_are_ignored():
    router, local = make_router(own=1)
    router.update_approx_tree(1, [5, 6])
    router.replace_approx_tree(1, [[5, 6]])
    assert 1 not in router.approx_trees
    assert local.total_tokens == 0


def test_replace_approx_tree():
    router, _ = make_router(own=0)
    router.update_approx_tree(2, [1, 2, 3])
    router.replace_approx_tree(2, [[4, 5]])
    assert router.match_lengths([1, 2, 3])[2] == 0
    assert router.match_lengths([4, 5])[2] == 2


def test_other_policies():
    rr, _ = make_router(own=1, policy="round_robin")
    assert [rr.route([1]) for _ in range(5)] == [0, 1, 2, 3, 0]
    me, _ = make_router(own=2, policy="self")
    assert me.route([1]) == 2


def test_mock_tokenize_is_stable_across_processes():
    code = "from src.router import mock_tokenize; print(mock_tokenize('the quick brown fox'))"
    outs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env={"PYTHONHASHSEED": seed}, check=True,
        ).stdout
        for seed in ("1", "2")
    }
    assert len(outs) == 1
    assert outs.pop().strip() == str(mock_tokenize("the quick brown fox"))
