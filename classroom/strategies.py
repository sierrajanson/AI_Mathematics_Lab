"""Teaching strategies: how a classroom of networks is trained.

All strategies share the same architecture, optimiser (Adam), learning rate and
number of epochs, so differences in outcome come from the strategy alone.

``lecture``
    Every student studies their own notes alone. Loss for student s is the MSE
    on their notes.

``shared_loss_2024``
    A faithful re-implementation of the 2024 "active learning" notebook: each
    group sums its members' individual losses and back-propagates the total.
    Because student s's weights appear only in its own term, the gradient each
    student receives is identical to studying alone (up to a constant scale
    that Adam normalises away). It is kept as a control to demonstrate this.

``group_study``
    Students in a group *discuss*: in addition to fitting their own notes, each
    student is pulled toward the average answer of their peers on every point
    the group has notes on (peers' answers are treated as fixed targets). This
    is Deep Mutual Learning (Zhang et al., 2018) adapted to regression.

``leader``
    Everyone studies alone for a warm-up period. Each group then picks as
    leader the student whose answers best match the *other* members' notes.
    Afterwards the leader keeps studying alone while the followers fit their
    own notes plus the leader's answers on the group's points. This is
    knowledge distillation (Hinton et al., 2015) with a teacher chosen from
    within the class.
"""
from dataclasses import dataclass

import torch

from .data import Lesson
from .model import Classroom

STRATEGIES = ["lecture", "shared_loss_2024", "group_study", "leader"]


@dataclass
class TrainConfig:
    group_size: int = 4
    epochs: int = 3000
    lr: float = 0.01
    peer_weight: float = 1.0      # lambda: weight of the peer / leader term
    leader_warmup: float = 1 / 3  # fraction of epochs studied alone before choosing leaders


def _group_pool(t: torch.Tensor, group_size: int) -> torch.Tensor:
    """(S, n, 1) -> (S, G*n, 1): each student gets the concatenated notes of its whole group."""
    s, n, _ = t.shape
    pooled = t.reshape(s // group_size, group_size * n, 1)
    return pooled.repeat_interleave(group_size, dim=0)


def _own_mse(model: Classroom, lesson: Lesson) -> torch.Tensor:
    """Per-student MSE on own notes, shape (S,)."""
    return ((model(lesson.notes_x) - lesson.notes_y) ** 2).mean(dim=(1, 2))


def _choose_leaders(model: Classroom, lesson: Lesson, group_size: int) -> torch.Tensor:
    """Index (into the class) of each group's leader: lowest MSE on the *other* members' notes."""
    s, n, _ = lesson.notes_x.shape
    g = s // group_size
    with torch.no_grad():
        pooled_x = _group_pool(lesson.notes_x, group_size)
        pooled_y = _group_pool(lesson.notes_y, group_size)
        err = (model(pooled_x) - pooled_y) ** 2  # (S, G*n, 1)
        err = err.reshape(g, group_size, group_size, n)  # [group, student, whose-notes, point]
        own = torch.eye(group_size, dtype=torch.bool)[None, :, :, None].expand_as(err)
        peer_err = err.masked_fill(own, 0).sum(dim=(2, 3)) / ((group_size - 1) * n)  # (g, group_size)
        local = peer_err.argmin(dim=1)
    return torch.arange(g) * group_size + local


def train(model: Classroom, lesson: Lesson, strategy: str, cfg: TrainConfig) -> dict:
    """Train ``model`` in place. Returns extra info (e.g. chosen leaders)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")
    s = model.n_students
    gs = cfg.group_size
    if s % gs:
        raise ValueError(f"{s} students cannot be split evenly into groups of {gs}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    pooled_x = _group_pool(lesson.notes_x, gs)
    warmup_end = int(cfg.epochs * cfg.leader_warmup)
    leaders = None
    info: dict = {}

    for epoch in range(cfg.epochs):
        own = _own_mse(model, lesson)

        if strategy == "lecture":
            loss = own.sum()

        elif strategy == "shared_loss_2024":
            # Sum within each group, then over groups, divided by class size -- as in 2024.
            loss = own.reshape(-1, gs).sum(dim=1).sum() / s

        elif strategy == "group_study":
            preds = model(pooled_x).reshape(s // gs, gs, -1)  # (groups, members, points)
            total = preds.sum(dim=1, keepdim=True)
            peer_mean = ((total - preds) / (gs - 1)).detach()  # average of the *other* members
            peer = ((preds - peer_mean) ** 2).mean(dim=2).reshape(s)
            loss = (own + cfg.peer_weight * peer).sum()

        else:  # leader
            if epoch < warmup_end:
                loss = own.sum()
            else:
                if leaders is None:
                    leaders = _choose_leaders(model, lesson, gs)
                    info["leaders"] = leaders.tolist()
                preds = model(pooled_x).squeeze(-1)  # (S, G*n)
                teacher = preds[leaders].detach().repeat_interleave(gs, dim=0)
                follow = ((preds - teacher) ** 2).mean(dim=1)
                is_leader = torch.zeros(s, dtype=torch.bool)
                is_leader[leaders] = True
                loss = (own + cfg.peer_weight * follow.masked_fill(is_leader, 0)).sum()

        opt.zero_grad()
        loss.backward()
        opt.step()

    return info
