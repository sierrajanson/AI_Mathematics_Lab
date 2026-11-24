"""The "concepts" a classroom is taught: three polynomials on [0, 1].

These are the same three target functions used throughout the original 2024
notebooks, so results stay comparable with that exploratory work.
"""
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class Concept:
    name: str
    formula: str
    fn: Callable[[torch.Tensor], torch.Tensor]


CONCEPTS = [
    Concept("arch", "4x - 4x^2", lambda x: 4 * x - 4 * x**2),
    Concept("parabola", "x^2", lambda x: x**2),
    Concept("wave", "x(3x - 2)^2", lambda x: x * (3 * x - 2) ** 2),
]


def concept_by_name(name: str) -> Concept:
    for c in CONCEPTS:
        if c.name == name:
            return c
    raise KeyError(f"Unknown concept {name!r}; choose from {[c.name for c in CONCEPTS]}")
