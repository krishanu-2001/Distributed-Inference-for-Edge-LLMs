"""SGLang backend path, exercised against a fake SGLang /generate server."""

import asyncio

from aiohttp import web

from src.config import Config
from src.network import NetworkSimulator
from src.node import InferenceNode, SGLangLLM, make_llm, MockLLM


def fake_sglang_app(cached_tokens: int, seen: list):
    async def generate(request):
        body = await request.json()
        seen.append(body)
        n = body["sampling_params"]["max_new_tokens"]
        return web.json_response({
            "text": "x",
            "output_ids": list(range(100, 100 + n)),
            "meta_info": {
                "prompt_tokens": len(body["input_ids"]),
                "completion_tokens": n,
                "cached_tokens": cached_tokens,
            },
        })
    app = web.Application()
    app.router.add_post("/generate", generate)
    return app


async def _serve(app, port):
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


def test_make_llm_selects_backend():
    cfg = Config()
    assert isinstance(make_llm(cfg, 0), MockLLM)
    cfg.llm.backend = "sglang"
    cfg.llm.sglang_base_port = 31000
    llm = make_llm(cfg, 2)
    assert isinstance(llm, SGLangLLM) and llm.url.endswith(":31002")


def test_node_records_sglang_actual_vs_expected_hit():
    async def run():
        seen = []
        server = await _serve(fake_sglang_app(cached_tokens=3, seen=seen), 31500)
        cfg = Config()
        cfg.llm.backend = "sglang"
        cfg.llm.sglang_base_port = 31500
        cfg.node.max_tokens = 1000
        node = InferenceNode(0, 31600, cfg, [0], {0: 31600}, NetworkSimulator(0, 0))
        try:
            first = await node._process_request([1, 2, 3, 4, 5], max_output_tokens=4)
            second = await node._process_request([1, 2, 3, 4, 9], max_output_tokens=4)
        finally:
            await node.llm.close()
            await server.cleanup()
        return seen, first, second, node

    seen, first, second, node = asyncio.run(run())
    assert seen[0]["input_ids"] == [1, 2, 3, 4, 5]
    assert seen[0]["sampling_params"] == {"max_new_tokens": 4, "temperature": 0.0, "ignore_eos": True}
    # actual hit comes from SGLang; expected from the node's own tree
    assert first["cache_hit_tokens"] == 3 and first["expected_hit_tokens"] == 0
    assert second["cache_hit_tokens"] == 3 and second["expected_hit_tokens"] == 4
    # SGLang's output ids are what gets cached locally
    assert node.radix_tree.match_prefix([1, 2, 3, 4, 5, 100, 101, 102, 103])[0] == 9
    assert node.radix_tree.total_tokens == node.kv_cache.used_tokens
