from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class ClusterConfig:
    num_nodes: int = 4
    base_port: int = 8200
    sglang_base_port: int = 9200


@dataclass
class NetworkConfig:
    lan_delay_ms: float = 2.0
    delay_jitter_ms: float = 0.5


@dataclass
class RouterConfig:
    policy: str = "cache_aware"
    routing_enabled: bool = True
    snapshot_poll_interval_s: float = 2.0
    local_snapshot_settle_timeout_s: float = 1.0
    local_snapshot_settle_poll_s: float = 0.05
    sync_tree_max_payload_mb: float = 64.0


@dataclass
class SGLangConfig:
    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    python_executable: str = "python"
    gpu_id: int | None = 0
    gpu_ids: list[int] = field(default_factory=list)
    tp_size: int = 1
    mem_fraction_static: float = 0.2
    context_length: int = 8192
    temperature: float = 0.0
    max_tokens: int = 32
    startup_timeout_s: float = 180.0
    enable_metrics: bool = True
    use_proxy_inference_time: bool = True
    log_dir: str = "runs/sglang_logs"
    use_local_checkout: bool = True

    def resolve_gpu_id(self, node_id: int) -> int | None:
        if self.gpu_ids:
            return self.gpu_ids[node_id % len(self.gpu_ids)]
        return self.gpu_id


@dataclass
class RequestsConfig:
    csv_path: str = "requests.csv"
    keep_running_after_demo: bool = False


@dataclass
class RunConfig:
    name: str | None = None
    overwrite_existing: bool = False


@dataclass
class Config:
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    requests: RequestsConfig = field(default_factory=RequestsConfig)
    run: RunConfig = field(default_factory=RunConfig)


def load_config(path: str = "config/sglang.yaml") -> Config:
    config = Config()
    config_path = Path(path)
    if not config_path.exists():
        return config

    with config_path.open() as handle:
        raw = yaml.safe_load(handle) or {}

    section_types = [
        ("cluster", ClusterConfig),
        ("network", NetworkConfig),
        ("router", RouterConfig),
        ("sglang", SGLangConfig),
        ("requests", RequestsConfig),
        ("run", RunConfig),
    ]
    for section_name, section_type in section_types:
        if section_name in raw:
            valid_keys = {field_info.name for field_info in fields(section_type)}
            section_values = {
                key: value
                for key, value in raw[section_name].items()
                if key in valid_keys
            }
            setattr(config, section_name, section_type(**section_values))

    return config
