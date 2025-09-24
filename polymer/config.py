from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = Field(default="mock")
    model: str = Field(default="")
    timeout_s: int = Field(default=60)


class MemoryConfig(BaseModel):
    data_dir: str = Field(default="./data")
    vector_db: str = Field(default="vector.db")
    episodes_db: str = Field(default="episodes.db")
    kg_file: str = Field(default="kg.graphml")
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5")


class ServerConfig(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)


class AlignmentConfig(BaseModel):
    max_tokens: int = Field(default=512)
    refuse_risky: bool = Field(default=True)
    blocked_topics: list[str] = Field(default_factory=list)


class PlannerConfig(BaseModel):
    max_depth: int = Field(default=4)
    max_branches: int = Field(default=3)


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)


def load_config(path: Optional[str] = None) -> AppConfig:
    config_path = Path(path or "config/default.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(**raw)


