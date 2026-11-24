## Teaching a Classroom of Neural Networks

*Lab for Human Mathematics · research March–June 2024, rebuilt and extended 2026*

![Screenshot 2024-05-25 132804](https://github.com/user-attachments/assets/93c63aaa-02bf-42c6-9d86-1351c300ae8b)

**Question:** if students were small neural networks, would they learn better alone, in
study groups, or following a group leader? And does the answer differ between reproducing
what they were taught and applying it somewhere new?

A "classroom" of 24 small networks (one hidden layer, 4–16 neurons each) learns a polynomial
from private, noisy notes. Each student is graded on their **homework** (own notes), a
**quiz** (new points in the range they studied, i.e. interpolation) and a **final exam** (a range
nobody studied, i.e. extrapolation). The three teaching strategies run on the *same* students, notes,
optimiser and study time, so the strategy is the only thing that changes:

| Strategy | What each student does | ML analogue |
|---|---|---|
| Lecture | fits their own notes alone | independent training |
| Group study | fits own notes **and** the average answer of their 3 groupmates | Deep Mutual Learning (Zhang et al., 2018) |
| Leader | after a solo warm-up, the student who best explains the others' notes leads; followers also fit the leader's answers | knowledge distillation (Hinton et al., 2015) |

### Findings

Class-average grades (0–100), averaged over 3 concepts and 5 independent classes:

| Strategy | Homework | Quiz (interpolation) | Final exam (extrapolation) |
|---|---|---|---|
| Lecture | **98.3** | 89.0 | 21.4 |
| Group study | 97.7 | **98.9** | **29.7** |
| Leader | 97.4 | 97.2 | 26.5 |
| *2024 "shared loss" (control)* | *98.3* | *89.0* | *21.4* |

- **Learning from classmates beats studying alone on the quiz in every class** (+9.9 points
  for group study, +8.3 for leader). It helps the weakest students most: under lecture, the
  bottom 10% of students score below 70, while under group study no student scores below 94.
- **Followers overtake their leader.** Leaders are chosen well (they'd score 95 alone vs. a
  class average of 89), but followers finish higher (97.9 vs. 95.4) by combining the leader's
  answers with their own notes.
- **Nobody learns to extrapolate.** Collaboration lowers variance on what the class has seen,
  but final-exam grades stay low for every strategy. The model itself limits extrapolation,
  not the teaching method.
- **The original 2024 "active learning" implementation never coupled the students.** Summing a group's losses gives
  each student exactly its solo gradient. It reproduces lecture learning to within 0.1 points;
  this is proven in the tests and shown in the notebook.

### Repository

| Path | Contents |
|---|---|
| [`classroom_experiments.ipynb`](classroom_experiments.ipynb) | **Start here.** Setup, strategies, results, figures and discussion |
| [`backprop_from_scratch.ipynb`](backprop_from_scratch.ipynb) | Backpropagation derived by hand, implemented in NumPy, verified against PyTorch autograd |
| [`classroom/`](classroom) | The simulation: batched classroom model, data, strategies, grading, experiment runner, plots |
| [`tests/`](tests) | Correctness tests (batched model = standalone networks, strategy coupling, leader choice, …) |
| [`results/students.csv`](results/students.csv) | One row per student per trial from the full run |
| [`archive/`](archive) | The original 2024 exploratory notebooks, with notes on known issues and attribution |

### Running it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest                       # tests (~30 s)
python -m classroom.experiment         # full experiment -> results/students.csv (~10-15 min on CPU)
python -m classroom.experiment --quick # 1-minute smoke test
jupyter notebook classroom_experiments.ipynb
```

Everything is seeded, so a rerun reproduces `results/students.csv` exactly on the same
software versions.
