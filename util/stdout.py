import os
import sys
from contextlib import contextmanager
from typing import Any


# Disable
def blockprint() -> None:
    sys.stdout = open(os.devnull, "w")


# Restore
def enableprint() -> None:
    sys.stdout = sys.__stdout__


@contextmanager
def suppress_stdout() -> Any:
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
