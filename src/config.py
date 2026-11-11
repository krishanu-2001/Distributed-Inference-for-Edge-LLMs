import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClusterConfig:
    num_nodes: int = 4
    base_port: int = 8100


@dataclass
class NetworkConfig:
    lan_delay_ms: float = 2.0
    delay_jitter_ms: float = 0.5


@dataclass
class NodeConfig:
    max_tokens: int = 262144
    eviction_policy: str = "lru"


@dataclass
class LLMConfig:
    backend: str = "mock"            # mock | sglang
    prefill_cost_factor: float = 0.0001   # mock only
    decode_cost_factor: float = 0.001     # mock only
    max_output_tokens: int = 20
    # sglang: node i talks to http://{sglang_host}:{sglang_base_port + i}
    sglang_host: str = "127.0.0.1"
    sglang_base_port: int = 30000
    request_timeout_s: float = 300.0


@dataclass
class RouterConfig:
    policy: str = "cache_aware"


@dataclass
class BroadcastConfig:
    trigger: str = "queue_length"
    queue_threshold: int = 3
    interval_s: float = 5.0


@dataclass
class RequestsConfig:
    path: str = "data/requests.csv"
    # round_robin | original | hash_topic | random | weighted (see request_loader.py)
    node_assignment: str = "round_robin"
    node_weights: list[float] | None = None  # for "weighted": one per node
    seed: int = 42


@dataclass
class Config:
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    node: NodeConfig = field(default_factory=NodeConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    broadcast: BroadcastConfig = field(default_factory=BroadcastConfig)
    requests: RequestsConfig = field(default_factory=RequestsConfig)


def load_config(path: str = "config/default.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        return Config()

    with open(p) as f:
        raw = yaml.safe_load(f)

    config = Config()
    if raw is None:
        return config

    for section_name, section_cls in [
        ("cluster", ClusterConfig),
        ("network", NetworkConfig),
        ("node", NodeConfig),
        ("llm", LLMConfig),
        ("router", RouterConfig),
        ("broadcast", BroadcastConfig),
        ("requests", RequestsConfig),
    ]:
        if section_name in raw:
            setattr(config, section_name, section_cls(**raw[section_name]))

    return config
