from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    id: str
    action: str
    inputs: dict = Field(default_factory=dict)
    outputs: dict = Field(default_factory=dict)
    assumptions: List[str] = Field(default_factory=list)
    verification: List[str] = Field(default_factory=list)
    children: List[str] = Field(default_factory=list)


class PlanGraph(BaseModel):
    steps: dict[str, PlanStep] = Field(default_factory=dict)
    root_id: Optional[str] = None
    uncertainty: dict = Field(default_factory=dict)


