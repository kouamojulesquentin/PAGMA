from pathlib import Path

from .bandgap import Bandgap
from .demo_rc import DemoRC
from .filter import Filter
from .opamp1 import Opamp1

REGISTRY = {c.name: c for c in (DemoRC, Opamp1, Filter, Bandgap)}


def get_circuit(name: str, circuits_dir="circuits"):
    if name not in REGISTRY:
        raise KeyError(f"Unknown Circuit: {name}. Available: {', '.join(REGISTRY)}")
    return REGISTRY[name](Path(circuits_dir))
