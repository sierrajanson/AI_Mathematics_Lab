"""Correctness checks for the classroom simulation. Run with: python -m pytest"""
import torch
from torch import nn

from classroom import CONCEPTS, Classroom, TrainConfig, make_lesson, train
from classroom.strategies import _choose_leaders, _group_pool


def _lesson(n_students=8, seed=0):
    return make_lesson(CONCEPTS[0], n_students, n_notes=5, noise=0.05, train_max=0.7,
                       generator=torch.Generator().manual_seed(seed))


def test_batched_student_matches_standalone_network():
    """Student s in the batch computes exactly what a plain nn.Sequential with its weights computes."""
    model = Classroom([3, 7, 5], generator=torch.Generator().manual_seed(0))
    x = torch.rand(10, 1)
    batched = model(x)
    for s in range(3):
        st = model.student_state(s)
        h = st["w1"].shape[1]
        net = nn.Sequential(nn.Linear(1, h), nn.Tanh(), nn.Linear(h, 1))
        with torch.no_grad():
            net[0].weight.copy_(st["w1"].T)
            net[0].bias.copy_(st["b1"][0])
            net[2].weight.copy_(st["w2"].T)
            net[2].bias.copy_(st["b2"][0])
        torch.testing.assert_close(batched[s], net(x))


def test_students_are_independent_under_lecture():
    """Changing one student's notes must not change any other student's training."""
    lesson_a = _lesson()
    lesson_b = _lesson()
    lesson_b.notes_y[0] += 1.0  # corrupt student 0's notes only
    cfg = TrainConfig(epochs=50)
    hidden = [4, 6, 8, 5, 4, 6, 8, 5]
    ma = Classroom(hidden, generator=torch.Generator().manual_seed(1))
    mb = Classroom(hidden, generator=torch.Generator().manual_seed(1))
    train(ma, lesson_a, "lecture", cfg)
    train(mb, lesson_b, "lecture", cfg)
    x = torch.rand(20, 1)
    pa, pb = ma(x), mb(x)
    assert not torch.allclose(pa[0], pb[0])
    torch.testing.assert_close(pa[1:], pb[1:])


def test_2024_shared_loss_gives_each_student_its_solo_gradient():
    """The core finding about the 2024 notebook: summing a group's losses does not couple students.

    Each student's gradient under the shared loss equals its solo gradient times 1/S.
    """
    lesson = _lesson()
    model = Classroom([5] * 8, generator=torch.Generator().manual_seed(2))
    own = ((model(lesson.notes_x) - lesson.notes_y) ** 2).mean(dim=(1, 2))

    shared = own.reshape(-1, 4).sum(dim=1).sum() / 8
    g_shared = torch.autograd.grad(shared, model.w1, retain_graph=True)[0]
    for s in range(8):
        g_solo = torch.autograd.grad(own[s], model.w1, retain_graph=True)[0]
        torch.testing.assert_close(g_shared[s], g_solo[s] / 8)


def test_2024_shared_loss_trains_like_lecture():
    """End to end: because Adam is scale-invariant, the 2024 strategy reproduces lecture learning."""
    lesson = _lesson()
    cfg = TrainConfig(epochs=200)
    hidden = [6] * 8
    a = Classroom(hidden, generator=torch.Generator().manual_seed(3))
    b = Classroom(hidden, generator=torch.Generator().manual_seed(3))
    train(a, lesson, "lecture", cfg)
    train(b, lesson, "shared_loss_2024", cfg)
    x = torch.rand(50, 1)
    torch.testing.assert_close(a(x), b(x), atol=1e-3, rtol=1e-3)


def test_group_study_actually_couples_students():
    """Unlike the 2024 version, group study lets one student's notes influence groupmates."""
    lesson_a, lesson_b = _lesson(), _lesson()
    lesson_b.notes_y[0] += 1.0
    cfg = TrainConfig(epochs=50)
    hidden = [6] * 8
    ma = Classroom(hidden, generator=torch.Generator().manual_seed(4))
    mb = Classroom(hidden, generator=torch.Generator().manual_seed(4))
    train(ma, lesson_a, "group_study", cfg)
    train(mb, lesson_b, "group_study", cfg)
    x = torch.rand(20, 1)
    pa, pb = ma(x), mb(x)
    assert not torch.allclose(pa[1], pb[1])        # groupmate affected
    torch.testing.assert_close(pa[4:], pb[4:])     # other group unaffected


def test_group_pool_layout():
    t = torch.arange(8 * 2, dtype=torch.float).reshape(8, 2, 1)  # student s has notes [2s, 2s+1]
    pooled = _group_pool(t, 4)
    assert pooled.shape == (8, 8, 1)
    assert pooled[0, :, 0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert pooled[3, :, 0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert pooled[4, :, 0].tolist() == [8, 9, 10, 11, 12, 13, 14, 15]


def test_leader_is_best_on_peers_notes():
    lesson = _lesson()
    model = Classroom([6] * 8, generator=torch.Generator().manual_seed(5))
    leaders = _choose_leaders(model, lesson, 4).tolist()
    assert leaders[0] in range(0, 4) and leaders[1] in range(4, 8)
    with torch.no_grad():
        for g, leader in enumerate(leaders):
            members = range(4 * g, 4 * g + 4)
            def peer_err(s):
                others = [m for m in members if m != s]
                x = lesson.notes_x[others].reshape(-1, 1)
                y = lesson.notes_y[others].reshape(-1, 1)
                return ((model(x)[s] - y) ** 2).mean().item()
            assert leader == min(members, key=peer_err)
