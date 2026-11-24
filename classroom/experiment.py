"""Run the lecture vs. group-study vs. leader comparison.

Fairness: within a (seed, concept) trial every strategy starts from the *same*
students (identical hidden sizes and initial weights) and the *same* notes.
Only the teaching strategy differs.

Usage::

    python -m classroom.experiment            # full run, writes results/
    python -m classroom.experiment --quick    # smoke test
"""
import argparse
import csv
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .concepts import CONCEPTS, Concept
from .data import make_lesson
from .grading import ASSESSMENTS, assess
from .model import Classroom, random_hidden_sizes
from .strategies import STRATEGIES, TrainConfig, train


@dataclass
class ExperimentConfig:
    n_students: int = 24
    n_notes: int = 8
    noise: float = 0.05
    train_max: float = 0.7
    hidden_low: int = 4
    hidden_high: int = 16
    seeds: list[int] = field(default_factory=lambda: list(range(5)))
    train: TrainConfig = field(default_factory=TrainConfig)


def run_trial(concept: Concept, concept_idx: int, seed: int, cfg: ExperimentConfig,
              strategies: tuple[str, ...] = tuple(STRATEGIES)) -> tuple[list[dict], dict]:
    """One class, one concept, every strategy. Returns per-student rows and trained models."""
    data_gen = torch.Generator().manual_seed(10_000 * seed + concept_idx)
    lesson = make_lesson(concept, cfg.n_students, cfg.n_notes, cfg.noise, cfg.train_max, data_gen)
    hidden = random_hidden_sizes(cfg.n_students, cfg.hidden_low, cfg.hidden_high, data_gen)
    init_seed = 20_000 * seed + concept_idx

    rows, models = [], {}
    for strategy in strategies:
        model = Classroom(hidden, generator=torch.Generator().manual_seed(init_seed))
        info = train(model, lesson, strategy, cfg.train)
        scores = assess(model, lesson)
        leaders = set(info.get("leaders", []))
        for s in range(cfg.n_students):
            row = {
                "seed": seed, "concept": concept.name, "strategy": strategy,
                "student": s, "group": s // cfg.train.group_size,
                "hidden": hidden[s], "is_leader": s in leaders,
            }
            row.update({k: round(v[s].item(), 6) for k, v in scores.items()})
            rows.append(row)
        models[strategy] = model
    return rows, {"lesson": lesson, "models": models}


def _trial_rows(args: tuple) -> tuple[list[dict], str]:
    concept_idx, seed, cfg = args
    torch.set_num_threads(1)  # tiny tensors: one thread per process is several times faster
    t0 = time.time()
    rows, _ = run_trial(CONCEPTS[concept_idx], concept_idx, seed, cfg)
    return rows, f"seed {seed}  {CONCEPTS[concept_idx].name:<8}  {time.time() - t0:5.1f}s"


def run_all(cfg: ExperimentConfig, verbose: bool = True, workers: int = 1) -> list[dict]:
    """Run every (seed, concept) trial. Results are identical for any ``workers``.

    ``workers > 1`` runs trials in separate processes, which helps on machines with
    several free cores (each process pays torch's import cost once).
    """
    jobs = [(ci, seed, cfg) for seed in cfg.seeds for ci in range(len(CONCEPTS))]
    rows = []
    if workers == 1:
        results = map(_trial_rows, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=min(workers, len(jobs), os.cpu_count() or 1))
        results = pool.map(_trial_rows, jobs)
    for trial_rows, msg in results:
        rows.extend(trial_rows)
        if verbose:
            print(msg, flush=True)
    if workers != 1:
        pool.shutdown()
    return rows


def summarize(rows: list[dict], metric: str = "grade") -> dict:
    """{(concept, strategy): {assessment: (mean, std)}} -- class average per seed, then mean/std over seeds.

    ``concept="all"`` pools the three concepts (averaged per seed first).
    """
    per_seed: dict = {}
    for r in rows:
        for a in ASSESSMENTS:
            for concept in (r["concept"], "all"):
                key = (concept, r["strategy"], a, r["seed"])
                per_seed.setdefault(key, []).append(r[f"{a}_{metric}"])
    out: dict = {}
    for (concept, strategy, a, _seed), vals in per_seed.items():
        out.setdefault((concept, strategy), {}).setdefault(a, []).append(statistics.fmean(vals))
    return {
        k: {a: (statistics.fmean(v), statistics.stdev(v) if len(v) > 1 else 0.0) for a, v in d.items()}
        for k, d in out.items()
    }


def paired_differences(rows: list[dict], baseline: str = "lecture", metric: str = "grade") -> dict:
    """Per-seed class-average difference vs. baseline, pooled over concepts.

    Returns {(strategy, assessment): (mean_diff, std_diff, n_seeds_better, n_seeds)}.
    Pairing by seed removes the shared variance from identical students and notes.
    """
    avg: dict = {}
    for r in rows:
        for a in ASSESSMENTS:
            avg.setdefault((r["strategy"], a, r["seed"], r["concept"]), []).append(r[f"{a}_{metric}"])
    avg = {k: statistics.fmean(v) for k, v in avg.items()}
    seeds = sorted({k[2] for k in avg})
    concepts = sorted({k[3] for k in avg})
    strategies = sorted({k[0] for k in avg})
    out = {}
    for strat in strategies:
        if strat == baseline:
            continue
        for a in ASSESSMENTS:
            diffs = [statistics.fmean(avg[(strat, a, s, c)] - avg[(baseline, a, s, c)] for c in concepts)
                     for s in seeds]
            out[(strat, a)] = (statistics.fmean(diffs),
                               statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
                               sum(d > 0 for d in diffs), len(diffs))
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("seed", "student", "group", "hidden"):
            r[k] = int(r[k])
        r["is_leader"] = r["is_leader"] == "True"
        for a in ASSESSMENTS:
            r[f"{a}_mse"] = float(r[f"{a}_mse"])
            r[f"{a}_grade"] = float(r[f"{a}_grade"])
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="2 seeds, 300 epochs (smoke test)")
    p.add_argument("--out", type=Path, default=Path("results/students.csv"))
    p.add_argument("--workers", type=int, default=1, help="parallel processes (default 1)")
    args = p.parse_args()

    cfg = ExperimentConfig()
    if args.quick:
        cfg.seeds = [0, 1]
        cfg.train.epochs = 300
    rows = run_all(cfg, workers=args.workers)
    write_csv(rows, args.out)

    summary = summarize(rows)
    print(f"\nClass-average grade (mean +/- std over {len(cfg.seeds)} seeds), all concepts:")
    print(f"{'strategy':<18}" + "".join(f"{a:>16}" for a in ASSESSMENTS))
    for strat in STRATEGIES:
        cells = summary[("all", strat)]
        print(f"{strat:<18}"
              + "".join(f"{cells[a][0]:>9.1f} +/- {cells[a][1]:<3.1f}" for a in ASSESSMENTS))
    print(f"\nWrote {len(rows)} rows to {args.out}")
    print("Config:", asdict(cfg))


if __name__ == "__main__":
    main()
