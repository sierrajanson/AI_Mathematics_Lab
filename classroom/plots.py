"""Figures for the classroom experiments (matplotlib, light theme).

Colour follows the strategy everywhere: lecture = blue, group study = orange,
leader = aqua (validated as a colour-blind-safe set). Every chart also carries a
legend or direct labels, and the notebook prints the same numbers as tables.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from .concepts import CONCEPTS

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SHADE = "#efeeea"
STRATEGY_COLORS = {"lecture": "#2a78d6", "group_study": "#eb6834", "leader": "#1baf7a"}
STRATEGY_LABELS = {"lecture": "Lecture (solo)", "group_study": "Group study", "leader": "Leader",
                   "shared_loss_2024": "Shared loss (2024)"}
ASSESSMENT_LABELS = {"homework": "Homework\n(own notes)", "quiz": "Quiz\n(interpolation)",
                     "exam": "Final exam\n(extrapolation)"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_2,
        "font.size": 9.5, "figure.dpi": 110,
    })


def plot_lesson(lesson_by_concept: dict, train_max: float, student: int = 0):
    """One student's notes for each concept, with the true curve and the unseen exam region."""
    fig, axes = plt.subplots(1, len(CONCEPTS), figsize=(11, 3.1), sharey=True)
    xs = torch.linspace(0, 1, 300)
    for ax, concept in zip(axes, CONCEPTS):
        lesson = lesson_by_concept[concept.name]
        ax.axvspan(train_max, 1, color=SHADE, lw=0, zorder=0)
        ax.plot(xs, concept.fn(xs), color=INK, lw=2, label="true concept")
        ax.scatter(lesson.notes_x[student, :, 0], lesson.notes_y[student, :, 0], s=36,
                   color=STRATEGY_COLORS["lecture"], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label="one student's notes")
        ax.set_title(f"{concept.name}:  y = {concept.formula}", loc="left")
        ax.set_xlabel("x")
        ax.text(train_max + 0.015, 0.97, "final-exam\nregion", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8.5, color=INK_2)
    axes[0].set_ylabel("y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", ncol=2, bbox_to_anchor=(0.01, 0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def plot_grades(summary: dict, strategies=("lecture", "group_study", "leader")):
    """Class-average grade per concept, one panel per assessment. Error bars: std over seeds."""
    assessments = list(ASSESSMENT_LABELS)
    groups = [c.name for c in CONCEPTS] + ["all"]
    fig, axes = plt.subplots(1, len(assessments), figsize=(11.5, 3.6), sharey=True)
    width = 0.26
    x = np.arange(len(groups))
    for ax, a in zip(axes, assessments):
        for i, strat in enumerate(strategies):
            means = [summary[(g, strat)][a][0] for g in groups]
            stds = [summary[(g, strat)][a][1] for g in groups]
            ax.bar(x + (i - 1) * width, means, width - 0.03, color=STRATEGY_COLORS[strat],
                   label=STRATEGY_LABELS[strat], zorder=2)
            ax.errorbar(x + (i - 1) * width, means, yerr=stds, fmt="none", ecolor=INK_2,
                        elinewidth=1, capsize=0, zorder=3)
            for xi, m in zip(x + (i - 1) * width, means):
                if m < 3:  # a zero-height bar would look like missing data
                    ax.text(xi, m + 1.5, f"{m:.0f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.set_xticks(x, [g if g != "all" else "all three" for g in groups])
        ax.set_title(ASSESSMENT_LABELS[a].replace("\n", " "), loc="left")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("class-average grade (0-100)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", ncol=3, bbox_to_anchor=(0.01, 0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def plot_group_predictions(trial: dict, group: int, group_size: int, train_max: float, concept,
                           strategies=("lecture", "group_study", "leader"), leaders=None):
    """What one study group learned under each strategy."""
    lesson, models = trial["lesson"], trial["models"]
    members = list(range(group * group_size, (group + 1) * group_size))
    xs = torch.linspace(0, 1, 300)[:, None]
    fig, axes = plt.subplots(1, len(strategies), figsize=(11, 3.3), sharey=True)
    for ax, strat in zip(axes, strategies):
        with torch.no_grad():
            preds = models[strat](xs)
        ax.axvspan(train_max, 1, color=SHADE, lw=0, zorder=0)
        for m in members:
            ax.scatter(lesson.notes_x[m, :, 0], lesson.notes_y[m, :, 0], s=14, color=INK_2,
                       alpha=0.55, linewidth=0, zorder=2)
        for m in members:
            is_lead = strat == "leader" and leaders is not None and m in leaders
            ax.plot(xs[:, 0], preds[m, :, 0], color=STRATEGY_COLORS[strat],
                    lw=2.6 if is_lead else 1.6, alpha=1 if is_lead else 0.85, zorder=3)
        ax.plot(xs[:, 0], concept.fn(xs[:, 0]), color=INK, lw=1.5, ls=(0, (4, 3)), zorder=4)
        ax.set_title(STRATEGY_LABELS[strat], loc="left", color=INK)
        ax.set_xlabel("x")
        ax.set_ylim(-0.25, 1.25)
    axes[0].set_ylabel("y")
    handles = [
        plt.Line2D([], [], color=INK, lw=1.5, ls=(0, (4, 3)), label="true concept"),
        plt.Line2D([], [], color=INK_2, marker="o", lw=0, ms=4, alpha=0.6, label="group's notes"),
        plt.Line2D([], [], color=INK_2, lw=1.6, label=f"each of the {group_size} students' answer"),
    ]
    if leaders is not None:
        handles.append(plt.Line2D([], [], color=STRATEGY_COLORS["leader"], lw=2.6, label="leader"))
    fig.legend(handles=handles, loc="lower left", ncol=len(handles), bbox_to_anchor=(0.01, 0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def plot_grade_spread(rows: list[dict], assessment: str = "quiz",
                      strategies=("lecture", "group_study", "leader")):
    """Every student's grade (all seeds and concepts) as a jittered strip, one row per strategy."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9, 2.9))
    for i, strat in enumerate(strategies):
        g = np.array([r[f"{assessment}_grade"] for r in rows if r["strategy"] == strat])
        y = len(strategies) - 1 - i + rng.uniform(-0.28, 0.28, len(g))
        ax.scatter(g, y, s=10, color=STRATEGY_COLORS[strat], alpha=0.45, linewidth=0)
        q10, med = np.percentile(g, 10), np.median(g)
        ax.plot([med, med], [len(strategies) - 1 - i - 0.36, len(strategies) - 1 - i + 0.36],
                color=INK, lw=2)
        ax.text(101.5, len(strategies) - 1 - i, f"median {med:.0f} · bottom 10% below {q10:.0f}",
                va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(strategies)), [STRATEGY_LABELS[s] for s in reversed(strategies)])
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"{assessment} grade of each student (black tick = median)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig
