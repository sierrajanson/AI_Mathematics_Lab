"""Lesson material for one class: each student's private notes plus shared tests.

* **Notes** (training data): every student independently writes down
  ``n_notes`` noisy observations of the concept, at random x in [0, train_max].
  Students see *different* points with *different* noise -- like real notes.
* **Quiz** (interpolation): fresh, noise-free points inside [0, train_max].
* **Final exam** (extrapolation): noise-free points in (train_max, 1], a region
  nobody took notes on. This is the "critical thinking" test: applying the
  concept beyond the examples seen in class.
"""
from dataclasses import dataclass

import torch

from .concepts import Concept


@dataclass
class Lesson:
    notes_x: torch.Tensor  # (S, n_notes, 1)
    notes_y: torch.Tensor  # (S, n_notes, 1)
    quiz_x: torch.Tensor   # (Nq, 1)
    quiz_y: torch.Tensor
    exam_x: torch.Tensor   # (Ne, 1)
    exam_y: torch.Tensor
    reference_var: float   # variance of the concept over [0, 1]; normalises grades


def make_lesson(
    concept: Concept,
    n_students: int,
    n_notes: int,
    noise: float,
    train_max: float,
    generator: torch.Generator,
    n_test: int = 200,
) -> Lesson:
    notes_x = torch.rand((n_students, n_notes, 1), generator=generator) * train_max
    notes_y = concept.fn(notes_x) + noise * torch.randn(notes_x.shape, generator=generator)

    quiz_x = torch.linspace(0, train_max, n_test)[:, None]
    exam_x = torch.linspace(train_max, 1, n_test + 1)[1:, None]
    full = concept.fn(torch.linspace(0, 1, 1001))
    return Lesson(
        notes_x=notes_x,
        notes_y=notes_y,
        quiz_x=quiz_x,
        quiz_y=concept.fn(quiz_x),
        exam_x=exam_x,
        exam_y=concept.fn(exam_x),
        reference_var=full.var().item(),
    )
