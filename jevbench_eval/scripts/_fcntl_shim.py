"""Windows shim for fcntl (not available on Windows).

JevBench uses fcntl.flock for its ledger. We don't actually need real file
locking because Windows file IO is already serialized at the OS level for the
single-writer case, and the harness runs serially by design.
"""
import sys as _sys
_FAKE_FCNTL = _sys.modules.get("fcntl") or type(_sys)("fcntl")
_FAKE_FCNTL.LOCK_EX = 2
_FAKE_FCNTL.LOCK_SH = 1
_FAKE_FCNTL.LOCK_UN = 8
_FAKE_FCNTL.LOCK_NB = 4
def _noop(*_a, **_kw):
    return None
_FAKE_FCNTL.flock = _noop
_FAKE_FCNTL.fcntl = _noop
_FAKE_FCNTL.ioctl = _noop
_FAKE_FCNTL.lockf = _noop
_sys.modules["fcntl"] = _FAKE_FCNTL
