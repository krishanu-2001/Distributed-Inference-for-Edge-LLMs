import asyncio
import random
import aiohttp


class NetworkSimulator:
    def __init__(self, lan_delay_ms: float, delay_jitter_ms: float):
        self.delay_s = lan_delay_ms / 1000.0
        self.jitter_s = delay_jitter_ms / 1000.0
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        """One shared connection pool instead of a new session per message."""
        if self._session is None or self._session.closed:
            # Generous total timeout: a forwarded request includes real inference.
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=600)
            )
        return self._session

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()

    def _get_delay(self) -> float:
        jitter = random.uniform(-self.jitter_s, self.jitter_s)
        return max(0, self.delay_s + jitter)

    async def simulate_delay(self):
        """Simulate one-way network delay."""
        await asyncio.sleep(self._get_delay())

    async def forward_request(
        self, to_port: int, token_ids: list[int], max_output_tokens: int | None = None
    ) -> dict:
        """Forward inference request to another node with round-trip delay."""
        # Outbound delay
        await self.simulate_delay()

        url = f"http://localhost:{to_port}/internal_infer"
        async with self._get_session().post(url, json={"token_ids": token_ids, "max_output_tokens": max_output_tokens}) as resp:
            result = await resp.json()

        # Return delay
        await self.simulate_delay()
        return result

    async def broadcast(
        self, from_port: int, all_ports: list[int], endpoint: str, data: dict
    ) -> list[dict]:
        """Broadcast data to all other nodes."""
        tasks = []
        for port in all_ports:
            if port == from_port:
                continue
            tasks.append(self._send_broadcast(port, endpoint, data))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if not isinstance(r, Exception)]
        return []

    async def _send_broadcast(self, port: int, endpoint: str, data: dict) -> dict:
        await self.simulate_delay()
        url = f"http://localhost:{port}/{endpoint}"
        async with self._get_session().post(url, json=data) as resp:
            result = await resp.json()

        await self.simulate_delay()
        return result
