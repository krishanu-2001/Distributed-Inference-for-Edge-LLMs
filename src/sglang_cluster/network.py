from __future__ import annotations

import asyncio
import random

import aiohttp


class NetworkSimulator:
    def __init__(self, lan_delay_ms: float, delay_jitter_ms: float):
        self.delay_s = lan_delay_ms / 1000.0
        self.jitter_s = delay_jitter_ms / 1000.0

    def _get_delay(self) -> float:
        jitter = random.uniform(-self.jitter_s, self.jitter_s)
        return max(0.0, self.delay_s + jitter)

    async def simulate_delay(self) -> float:
        delay_s = self._get_delay()
        await asyncio.sleep(delay_s)
        return delay_s

    async def post_json(self, to_port: int, endpoint: str, data: dict) -> dict:
        await self.simulate_delay()
        url = f"http://127.0.0.1:{to_port}/{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(
                        f"POST {url} failed with HTTP {response.status}: "
                        f"{body[:200]}"
                    )
                payload = await response.json()
        await self.simulate_delay()
        return payload

    async def forward_request(self, to_port: int, payload: dict) -> tuple[dict, float]:
        communication_time_s = await self.simulate_delay()
        url = f"http://127.0.0.1:{to_port}/internal_infer"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                forwarded_payload = await response.json()
        communication_time_s += await self.simulate_delay()
        return forwarded_payload, communication_time_s

    async def broadcast(
        self, from_port: int, all_ports: list[int], endpoint: str, data: dict
    ) -> list[dict]:
        tasks = []
        for port in all_ports:
            if port == from_port:
                continue
            tasks.append(self.post_json(port, endpoint, data))

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [result for result in results if not isinstance(result, Exception)]
