from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Constraint(BaseModel):
    name: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)


class Preference(BaseModel):
    name: str
    value: str
    weight: float = Field(ge=0.0, le=1.0, default=0.5)


class IntentFrame(BaseModel):
    intent: str
    roles: dict[str, str] = Field(default_factory=dict)
    constraints: List[Constraint] = Field(default_factory=list)
    preferences: List[Preference] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)


class IntentionGraph(BaseModel):
    frames: List[IntentFrame] = Field(default_factory=list)
    alternatives: List[IntentFrame] = Field(default_factory=list)
    context: dict = Field(default_factory=dict)


