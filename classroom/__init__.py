"""Simulating teaching strategies with a classroom of small neural networks."""
from .concepts import CONCEPTS, Concept, concept_by_name
from .data import Lesson, make_lesson
from .grading import ASSESSMENTS, assess
from .model import Classroom, random_hidden_sizes
from .strategies import STRATEGIES, TrainConfig, train

__all__ = [
    "ASSESSMENTS", "CONCEPTS", "Classroom", "Concept", "Lesson", "STRATEGIES",
    "TrainConfig", "assess", "concept_by_name", "make_lesson", "random_hidden_sizes", "train",
]
