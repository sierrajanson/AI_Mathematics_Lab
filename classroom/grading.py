"""Grading a trained classroom.

Each student gets an MSE on three assessments (homework = own noisy notes,
quiz = interpolation, final exam = extrapolation) and a 0-100 grade:

    grade = 100 * clip(1 - MSE / Var[concept over [0, 1]], 0, 1)

i.e. the R^2 score against a fixed reference variance, so a grade of 0 means
"no better than always answering the concept's average value" and the three
assessments are on the same scale. This replaces the hand-tuned
``calc_letter_grade`` lookup from the 2024 notebook.
"""
import torch

from .data import Lesson
from .model import Classroom

ASSESSMENTS = ["homework", "quiz", "exam"]


def grade(mse: torch.Tensor, reference_var: float) -> torch.Tensor:
    return 100 * (1 - mse / reference_var).clamp(0, 1)


@torch.no_grad()
def assess(model: Classroom, lesson: Lesson) -> dict[str, torch.Tensor]:
    """Per-student MSE and grade for each assessment. Each value has shape (S,)."""
    mse = {
        "homework": ((model(lesson.notes_x) - lesson.notes_y) ** 2).mean(dim=(1, 2)),
        "quiz": ((model(lesson.quiz_x) - lesson.quiz_y) ** 2).mean(dim=(1, 2)),
        "exam": ((model(lesson.exam_x) - lesson.exam_y) ** 2).mean(dim=(1, 2)),
    }
    out = {f"{k}_mse": v for k, v in mse.items()}
    out.update({f"{k}_grade": grade(v, lesson.reference_var) for k, v in mse.items()})
    return out
