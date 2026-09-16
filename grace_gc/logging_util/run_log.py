"""Tee stdout/stderr into run.log so a crash still leaves the console trail."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path


class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for fh in self.files:
            fh.write(data)
            fh.flush()
        return len(data)

    def flush(self):
        for fh in self.files:
            fh.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.files[0], name)


class RunLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = _Tee(self._stdout, self.fh)
        sys.stderr = _Tee(self._stderr, self.fh)

    def close(self) -> None:
        try:
            sys.stdout = self._stdout
            sys.stderr = self._stderr
        finally:
            self.fh.close()

    def __enter__(self) -> RunLog:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            traceback.print_exception(exc_type, exc, tb)
        self.close()
        return False
