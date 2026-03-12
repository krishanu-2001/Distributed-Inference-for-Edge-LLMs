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
    prefill_cost_factor: float = 0.0001
    decode_cost_factor: float = 0.001
    max_output_tokens: int = 20


@dataclass
class RouterConfig:
    policy: str = "cache_aware"


@dataclass
class BroadcastConfig:
    trigger: str = "queue_length"
    queue_threshold: int = 3
    interval_s: float = 5.0


@dataclass
class Config:
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    node: NodeConfig = field(default_factory=NodeConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    broadcast: BroadcastConfig = field(default_factory=BroadcastConfig)


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
    ]:
        if section_name in raw:
            setattr(config, section_name, section_cls(**raw[section_name]))

    return config
