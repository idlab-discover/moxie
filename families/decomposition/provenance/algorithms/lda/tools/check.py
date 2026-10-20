"""Fail closed until LDA implementation and acceptance checks are supplied."""
import sys

if __name__ == "__main__":
    action = sys.argv[1]
    if action not in ("build", "test", "parity"):
        raise ValueError(action)
    sys.exit(f"BLOCKED: {action}: LatentDirichletAllocation is not implemented; "
             "shared RandomState.gamma is missing. Run pixi run reproduce for evidence.")
