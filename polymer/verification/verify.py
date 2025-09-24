from __future__ import annotations

from typing import Dict, List


def quick_checks(answer: str) -> List[str]:
    issues: List[str] = []
    if len(answer.strip()) == 0:
        issues.append("empty_answer")
    if any(len(line) > 2000 for line in answer.splitlines()):
        issues.append("line_too_long")
    return issues

def unit_assertions(assertions: List[str]) -> List[str]:
    # Placeholder: evaluate truthy/falsey expressions like '2+2==4'
    issues: List[str] = []
    for expr in assertions:
        try:
            if not bool(eval(expr, {"__builtins__": {}})):
                issues.append(f"assertion_failed:{expr}")
        except Exception:
            issues.append(f"assertion_error:{expr}")
    return issues


