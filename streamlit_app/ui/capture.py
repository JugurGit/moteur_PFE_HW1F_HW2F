from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

def capture_stdout(fn, *args, **kwargs) -> tuple[object, str]:
    buf = StringIO()
    with redirect_stdout(buf):
        out = fn(*args, **kwargs)
    return out, buf.getvalue()
