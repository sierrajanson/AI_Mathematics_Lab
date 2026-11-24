"""A whole classroom of small neural networks, stored as one batched module.

Every student is a 1 -> H -> 1 tanh network. Students are stacked along a
leading "student" dimension so the class trains in a single vectorised pass
instead of a Python loop over separate ``nn.Module`` objects.

Batching does not couple the students: each student only ever touches its own
slice of each weight tensor, and Adam's update is element-wise, so one Adam
optimiser over the batched tensors is exactly equivalent to one optimiser per
student (see ``tests/test_classroom.py``).

Students can differ in capacity: student ``s`` uses only its first
``hidden_sizes[s]`` hidden units, mirroring the 2024 notebooks where each
student got a random number of neurons.
"""
import torch
from torch import nn


class Classroom(nn.Module):
    def __init__(self, hidden_sizes: list[int], generator: torch.Generator | None = None):
        super().__init__()
        n = len(hidden_sizes)
        h_max = max(hidden_sizes)
        sizes = torch.tensor(hidden_sizes)
        mask = (torch.arange(h_max)[None, :] < sizes[:, None]).float()  # (S, H)

        def uniform(shape, bound):
            return (torch.rand(shape, generator=generator) * 2 - 1) * bound

        # PyTorch nn.Linear default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        out_bound = (1.0 / sizes.float().sqrt())[:, None, None]  # fan_in = student's hidden size
        self.w1 = nn.Parameter(uniform((n, 1, h_max), 1.0))
        self.b1 = nn.Parameter(uniform((n, 1, h_max), 1.0))
        self.w2 = nn.Parameter(uniform((n, h_max, 1), 1.0) * out_bound)
        self.b2 = nn.Parameter(uniform((n, 1, 1), 1.0) * out_bound)
        self.register_buffer("mask", mask[:, None, :])  # (S, 1, H)
        self.hidden_sizes = list(hidden_sizes)

    @property
    def n_students(self) -> int:
        return self.w1.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (S, N, 1) -- each student gets its own batch of N inputs. Returns (S, N, 1)."""
        if x.dim() == 2:  # same inputs for everyone: (N, 1) -> (S, N, 1)
            x = x.expand(self.n_students, *x.shape)
        h = torch.tanh(x @ self.w1 + self.b1) * self.mask
        return h @ self.w2 + self.b2

    def student_state(self, s: int) -> dict[str, torch.Tensor]:
        """Weights of one student, trimmed to its real hidden size (for inspection/tests)."""
        h = self.hidden_sizes[s]
        return {
            "w1": self.w1[s, :, :h].detach().clone(),
            "b1": self.b1[s, :, :h].detach().clone(),
            "w2": self.w2[s, :h, :].detach().clone(),
            "b2": self.b2[s].detach().clone(),
        }


def random_hidden_sizes(n_students: int, low: int, high: int, generator: torch.Generator) -> list[int]:
    """Hidden-layer size for each student, uniform in [low, high]."""
    return torch.randint(low, high + 1, (n_students,), generator=generator).tolist()
