import os
import sys

# Disable
def blockprint() -> None:
    sys.stdout = open(os.devnull, "w")


# Restore
def enableprint() -> None:
    sys.stdout = sys.__stdout__
