from __future__ import annotations

from typing import Any, Dict
from RestrictedPython import compile_restricted
from RestrictedPython import safe_builtins
from RestrictedPython.Eval import default_guarded_getitem
from RestrictedPython.Guards import guarded_iter_unpack_sequence, guarded_unpack_sequence


class ProgramSandbox:
    def __init__(self) -> None:
        pass

    def run(self, code: str, inputs: Dict[str, Any] | None = None) -> Dict[str, Any]:
        byte_code = compile_restricted(code, filename="<pot>", mode="exec")
        env = {
            "__builtins__": safe_builtins,
            "_getattr_": getattr,
            "_getitem_": default_guarded_getitem,
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            "_unpack_sequence_": guarded_unpack_sequence,
        }
        env.update(inputs or {})
        exec(byte_code, env)
        # Collect variables except internals
        return {k: v for k, v in env.items() if not k.startswith("_")}


