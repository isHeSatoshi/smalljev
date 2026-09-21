"""Windows compatibility shim for `fcntl` (POSIX file locking).

JevBench's `budget.py` imports `fcntl` for cross-process ledger locking. On Windows
`fcntl` does not exist, but our benchmark is single-process and the locking is
therefore redundant. This module provides a no-op shim with the same surface area
that `jevbench/budget.py` calls into.

Installed by `scripts/run_variant.py` and `scripts/summarize_run.py` before any
JevBench import. NOT used outside this isolated evaluation directory.
"""
import sys
import types


class _FlockModule(types.ModuleType):
    """Stub fcntl: implements LOCK_SH / LOCK_EX / LOCK_UN as no-ops."""

    LOCK_SH = 1
    LOCK_EX = 2
    LOCK_UN = 8
    LOCK_NB = 4

    def flock(self, f, op):  # noqa: D401
        return None


shim = _FlockModule("fcntl")
sys.modules["fcntl"] = shim