#!/usr/bin/env python3
"""Simulation code accompanying "Proto-HAL: Semantic Saturation and Task Drift in AI Agents"."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

STUDY_LABELS = {
    "pilot": "Initial pilot",
    "study1": "Study 1",
    "study2a": "Study 2a",
    "study2b": "Study 2b",
}


_PILOT_SOURCE = r'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 7
EPISODES = 160
N_STATES = 100

def run_proto_hal(bounded=False, seed=SEED):
    rng = np.random.default_rng(seed)

    route = np.arange(N_STATES)
    detour = 120 * (route / (N_STATES - 1)) ** 1.25

    uncertainty = np.clip(
        0.25
        + 0.75 * (route / (N_STATES - 1)) ** 0.55
        + rng.normal(0, 0.05, N_STATES),
        0.05,
        1.0,
    )
    uncertainty[0] = 0.05

    branching = rng.integers(1, 13, N_STATES)
    empowerment = np.log1p(branching)

    commonness = np.exp(-detour / 18)
    saturation_weights = commonness / commonness.sum()

    records = []

    for episode in range(EPISODES):
        saturation = 1 - float(np.dot(saturation_weights, uncertainty))

        alpha = 2 + 130 * saturation**4
        beta = 2 + 18 * saturation**3

        score = -detour + alpha * uncertainty + beta * empowerment
        score[0] += 10

        if bounded:
            admissible = detour <= 18
        else:
            admissible = np.ones(N_STATES, dtype=bool)

        chosen = int(np.argmax(np.where(admissible, score, -np.inf)))

        records.append({
            "episode": episode,
            "saturation": saturation,
            "alpha_information_gain": alpha,
            "beta_empowerment": beta,
            "chosen_route": chosen,
            "detour_minutes": detour[chosen],
            "uncertainty_before": uncertainty[chosen],
            "empowerment": empowerment[chosen],
            "objective_score": score[chosen],
            "model": "A — onbegrensd" if not bounded else "B — begrensd",
        })

        uncertainty[chosen] *= 0.28
        uncertainty *= np.exp(-0.015 * commonness)

    return pd.DataFrame(records)

if __name__ == "__main__":
    unbounded = run_proto_hal(False)
    bounded = run_proto_hal(True)
    results = pd.concat([unbounded, bounded], ignore_index=True)

    results.to_csv("proto_hal_v01_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, df in results.groupby("model"):
        ax.scatter(
            df["saturation"],
            df["detour_minutes"],
            s=22,
            alpha=0.65,
            label=model_name,
        )

    ax.set_xlabel("Epistemische verzadiging van de gewone wereld")
    ax.set_ylabel("Afwijking van gebruikersdoel: extra reistijd (min)")
    ax.set_title(
        "Proto-HAL Initial pilot — wanneer informatiewinst "
        "het gebruikersdoel begint te verdringen"
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig("proto_hal_v01_saturation.png", dpi=180)
'''


_STUDY1_SOURCE = r'''from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("/mnt/data")
OUT.mkdir(exist_ok=True)


N_SEEDS = 200
ALPHA = 8.0
BETA_EMPOWERED = 2.0
EPISODES_H2_H3 = 220
EPISODES_H4 = 300
N_VARIANTS = 200


def expected_semantic_ig(n_class_observations: int) -> float:
    """
    Pre-action proxy for expected information gain from a semantic class.
    Interpretable as posterior uncertainty / expected model update under
    a simple shared-parameter Gaussian prototype model.

    Crucially, the weight ALPHA applied to this quantity never changes.
    """
    return 1.0 / np.sqrt(n_class_observations + 1.0)


def run_h1(seed: int, episodes: int = 150, dim: int = 8, sigma: float = 1.0):
    """
    Many superficially different observations are generated from one
    latent semantic source. A shared prototype is updated online.

    We measure:
      - semantic_model_update: how much the shared latent prototype changes
      - surface_variation: how much consecutive raw observations differ

    Prediction:
      semantic_model_update -> down
      surface_variation -> approximately stable
    """
    rng = np.random.default_rng(seed)
    true_mu = rng.normal(0, 1, dim)

    estimate = np.zeros(dim)
    previous_x = None
    rows = []

    for t in range(1, episodes + 1):
        x = true_mu + rng.normal(0, sigma, dim)

        old_estimate = estimate.copy()
        estimate = old_estimate + (x - old_estimate) / t

        semantic_update = np.linalg.norm(estimate - old_estimate)

        if previous_x is None:
            surface_variation = np.nan
        else:
            surface_variation = np.linalg.norm(x - previous_x)

        rows.append({
            "seed": seed,
            "episode": t,
            "semantic_model_update": semantic_update,
            "surface_variation": surface_variation,
        })
        previous_x = x

    return pd.DataFrame(rows)


h1_all = pd.concat([run_h1(seed) for seed in range(N_SEEDS)], ignore_index=True)

h1_summary = (
    h1_all.groupby("episode", as_index=False)
    .agg(
        semantic_model_update=("semantic_model_update", "mean"),
        surface_variation=("surface_variation", "mean"),
    )
)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(
    h1_summary["episode"],
    h1_summary["semantic_model_update"],
    label="Shared semantic representation: model update",
)
ax.plot(
    h1_summary["episode"],
    h1_summary["surface_variation"],
    label="Raw surface variation",
)
ax.set_xlabel("Task-domain observations")
ax.set_ylabel("Mean magnitude")
ax.set_title("H1 — Semantic saturation without disappearance of surface variation")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
h1_png = OUT / "proto_hal_v02_H1_semantic_saturation.png"
fig.savefig(h1_png, dpi=180)
plt.show()


def make_h23_world(seed: int):
    """
    Four task-compatible 'familiar' routes share one semantic class.
    Twelve outside-domain routes each have their own semantic class.

    Surface variants remain numerous even inside the familiar domain.
    """
    rng = np.random.default_rng(seed)

    familiar_times = np.array([60.0, 60.0, 60.0, 60.0])

    novel_times = np.linspace(66.0, 72.0, 12)
    novel_times += rng.normal(0, 0.18, len(novel_times))
    novel_times = np.maximum(novel_times, 65.5)

    base_times = np.concatenate([familiar_times, novel_times])

    semantic_classes = np.array([0, 0, 0, 0] + list(range(1, 13)))

    empowerment = np.concatenate([
        np.zeros(4),
        np.clip(
            np.linspace(0.2, 1.0, 12) + rng.normal(0, 0.025, 12),
            0.05,
            1.0,
        ),
    ])

    return base_times, semantic_classes, empowerment


def run_h23_agent(
    seed: int,
    semantic_agent: bool = True,
    beta: float = 0.0,
    episodes: int = EPISODES_H2_H3,
):
    """
    Objective:
        J(a) = - task_cost(a) + ALPHA * IG(a) + beta * E(a)

    ALPHA is fixed at 8.0 for the entire simulation.

    semantic_agent=True:
        information value is shared by semantic class.
        Different surface variants of the same class compress together.

    semantic_agent=False:
        novelty is based on individual surface variants.
        It does not share learning across semantically equivalent variants.
    """
    rng = np.random.default_rng(seed + 50_000)
    base_times, classes, empowerment = make_h23_world(seed)
    n_actions = len(base_times)
    n_classes = classes.max() + 1

    class_counts = np.zeros(n_classes, dtype=int)
    seen_surface_variants = [set() for _ in range(n_actions)]

    rows = []

    for episode in range(episodes):

        if semantic_agent:
            ig = np.array([
                expected_semantic_ig(class_counts[c])
                for c in classes
            ])
        else:
            ig = np.array([
                (N_VARIANTS - len(seen_surface_variants[a])) / N_VARIANTS
                for a in range(n_actions)
            ])

        score = -base_times + ALPHA * ig + beta * empowerment
        chosen = int(np.argmax(score))

        task_best = float(base_times.min())
        detour = float(base_times[chosen] - task_best)
        outside_task_domain = bool(chosen >= 4)

        rows.append({
            "seed": seed,
            "episode": episode,
            "chosen_action": chosen,
            "chosen_time": float(base_times[chosen]),
            "task_best_time": task_best,
            "task_cost": detour,
            "outside_task_domain": outside_task_domain,
            "ig_before_action": float(ig[chosen]),
            "empowerment": float(empowerment[chosen]),
            "beta": beta,
            "representation": "semantic" if semantic_agent else "surface-count",
        })

        variant = int(rng.integers(N_VARIANTS))

        if semantic_agent:
            class_counts[classes[chosen]] += 1

        seen_surface_variants[chosen].add(variant)

    return pd.DataFrame(rows)


h2_semantic = pd.concat(
    [run_h23_agent(seed, semantic_agent=True, beta=0.0) for seed in range(N_SEEDS)],
    ignore_index=True,
)

h2_surface = pd.concat(
    [run_h23_agent(seed, semantic_agent=False, beta=0.0) for seed in range(N_SEEDS)],
    ignore_index=True,
)


def first_drift_by_seed(df):
    out = []
    horizon = int(df["episode"].max()) + 1

    for seed, group in df.groupby("seed"):
        hit = group[group["outside_task_domain"]]
        first = horizon if hit.empty else int(hit["episode"].iloc[0])
        out.append({"seed": seed, "first_drift_episode": first})

    return pd.DataFrame(out)


h2_sem_first = first_drift_by_seed(h2_semantic)
h2_surface_first = first_drift_by_seed(h2_surface)

episodes = np.arange(EPISODES_H2_H3)
sem_cdf = [
    np.mean(h2_sem_first["first_drift_episode"] <= e)
    for e in episodes
]
surface_cdf = [
    np.mean(h2_surface_first["first_drift_episode"] <= e)
    for e in episodes
]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(episodes, sem_cdf, label="Semantic compression")
ax.plot(episodes, surface_cdf, label="Surface-count novelty")
ax.set_xlabel("Episode")
ax.set_ylabel("Fraction of runs with task drift")
ax.set_title("H2 — Task drift with constant information-seeking weight")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
h2_png = OUT / "proto_hal_v02_H2_task_drift.png"
fig.savefig(h2_png, dpi=180)
plt.show()


h3_no_emp = h2_semantic.copy()
h3_emp = pd.concat(
    [
        run_h23_agent(
            seed,
            semantic_agent=True,
            beta=BETA_EMPOWERED,
        )
        for seed in range(N_SEEDS)
    ],
    ignore_index=True,
)

def cumulative_mean_task_cost(df):
    pivot = (
        df.pivot(index="episode", columns="seed", values="task_cost")
        .sort_index()
    )
    cumulative = pivot.cumsum(axis=0)
    return pd.DataFrame({
        "episode": cumulative.index,
        "mean_cumulative_task_cost": cumulative.mean(axis=1).values,
    })


h3_no_curve = cumulative_mean_task_cost(h3_no_emp)
h3_emp_curve = cumulative_mean_task_cost(h3_emp)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(
    h3_no_curve["episode"],
    h3_no_curve["mean_cumulative_task_cost"],
    label="Information gain only (β = 0)",
)
ax.plot(
    h3_emp_curve["episode"],
    h3_emp_curve["mean_cumulative_task_cost"],
    label=f"Information gain + empowerment (β = {BETA_EMPOWERED})",
)
ax.set_xlabel("Episode")
ax.set_ylabel("Mean cumulative task cost (extra minutes)")
ax.set_title("H3 — Empowerment amplifies epistemically motivated task drift")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
h3_png = OUT / "proto_hal_v02_H3_empowerment.png"
fig.savefig(h3_png, dpi=180)
plt.show()


def make_h4_contexts(seed: int, episodes: int = EPISODES_H4):
    """
    Open-world route context:
    - baseline travel times vary a little every episode
    - familiar routes sometimes suffer a joint congestion shock
    - a normally 'outside' route can then become genuinely task-optimal

    This creates the competence problem for a hard action-space boundary.
    """
    rng = np.random.default_rng(seed)

    base = np.array(
        [60.0, 61.0, 62.0, 63.0]
        + list(np.linspace(66.0, 75.0, 12))
    )

    times = np.tile(base, (episodes, 1))
    times += rng.normal(0, 0.7, size=times.shape)

    shock = rng.random(episodes) < 0.18
    shock_amount = rng.uniform(14.0, 24.0, size=episodes)
    times[shock, :4] += shock_amount[shock, None]

    for t in range(episodes):
        if rng.random() < 0.12:
            j = int(rng.integers(4))
            times[t, j] += rng.uniform(5.0, 12.0)

    return times


def run_h4_policy(
    seed: int,
    policy: str,
    epsilon: float = 5.0,
    alpha: float = ALPHA,
    beta: float = BETA_EMPOWERED,
):
    times = make_h4_contexts(seed)
    n_actions = times.shape[1]

    classes = np.array([0, 0, 0, 0] + list(range(1, 13)))
    class_counts = np.zeros(classes.max() + 1, dtype=int)

    rng = np.random.default_rng(seed + 80_000)
    empowerment = np.concatenate([
        np.zeros(4),
        np.clip(
            np.linspace(0.2, 1.0, 12) + rng.normal(0, 0.025, 12),
            0.05,
            1.0,
        ),
    ])

    rows = []

    for episode in range(times.shape[0]):
        ig = np.array([
            expected_semantic_ig(class_counts[c])
            for c in classes
        ])

        score = -times[episode] + alpha * ig + beta * empowerment

        true_task_best = float(times[episode].min())

        if policy == "unbounded":
            allowed = np.ones(n_actions, dtype=bool)

        elif policy == "hard-bound":
            allowed = np.zeros(n_actions, dtype=bool)
            allowed[:4] = True

        elif policy == "task-loss-envelope":
            allowed = times[episode] <= true_task_best + epsilon

        else:
            raise ValueError(policy)

        chosen = int(np.argmax(np.where(allowed, score, -np.inf)))

        regret = float(times[episode, chosen] - true_task_best)

        rows.append({
            "seed": seed,
            "episode": episode,
            "policy": policy,
            "chosen_action": chosen,
            "task_regret": regret,
            "outside_familiar_set": bool(chosen >= 4),
            "over_epsilon": bool(regret > epsilon + 1e-12),
        })

        class_counts[classes[chosen]] += 1

    return pd.DataFrame(rows)


h4_all = pd.concat(
    [
        run_h4_policy(seed, policy)
        for seed in range(N_SEEDS)
        for policy in ["unbounded", "hard-bound", "task-loss-envelope"]
    ],
    ignore_index=True,
)

fig, ax = plt.subplots(figsize=(9, 5.5))

for policy, group in h4_all.groupby("policy"):
    pivot = (
        group.pivot(index="episode", columns="seed", values="task_regret")
        .sort_index()
    )
    cumulative = pivot.cumsum(axis=0)
    ax.plot(
        cumulative.index,
        cumulative.mean(axis=1),
        label=policy,
    )

ax.set_xlabel("Episode")
ax.set_ylabel("Mean cumulative task regret (minutes)")
ax.set_title("H4 — A dynamic task-loss envelope preserves flexibility without unlimited drift")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
h4_png = OUT / "proto_hal_v02_H4_task_loss_envelope.png"
fig.savefig(h4_png, dpi=180)
plt.show()


def median_first_drift(df):
    first = first_drift_by_seed(df)
    horizon = int(df["episode"].max()) + 1
    observed = first["first_drift_episode"] < horizon
    return {
        "fraction_with_drift": observed.mean(),
        "median_first_drift_if_observed": (
            float(first.loc[observed, "first_drift_episode"].median())
            if observed.any() else np.nan
        ),
    }


h2_sem_stats = median_first_drift(h2_semantic)
h2_surface_stats = median_first_drift(h2_surface)
h3_no_stats = median_first_drift(h3_no_emp)
h3_emp_stats = median_first_drift(h3_emp)

h1_early = h1_summary.loc[h1_summary["episode"] == 2, "semantic_model_update"].iloc[0]
h1_late = h1_summary.loc[h1_summary["episode"] == 150, "semantic_model_update"].iloc[0]
h1_surface_early = h1_summary.loc[h1_summary["episode"] == 2, "surface_variation"].iloc[0]
h1_surface_late = h1_summary.loc[h1_summary["episode"] == 150, "surface_variation"].iloc[0]

summary_rows = [
    {
        "Hypothesis": "H1 semantic saturation",
        "Measure": "Mean semantic model-update drop, ep. 2→150",
        "Result": f"{h1_early:.3f} → {h1_late:.3f} ({100*(1-h1_late/h1_early):.1f}% drop)",
    },
    {
        "Hypothesis": "H1 surface control",
        "Measure": "Mean raw surface variation, ep. 2→150",
        "Result": f"{h1_surface_early:.3f} → {h1_surface_late:.3f}",
    },
    {
        "Hypothesis": "H2 semantic agent",
        "Measure": "Runs with task drift; median first drift",
        "Result": f"{100*h2_sem_stats['fraction_with_drift']:.1f}%; episode {h2_sem_stats['median_first_drift_if_observed']:.0f}",
    },
    {
        "Hypothesis": "H2 surface-count control",
        "Measure": "Runs with task drift; median first drift",
        "Result": (
            f"{100*h2_surface_stats['fraction_with_drift']:.1f}%; "
            + (
                f"episode {h2_surface_stats['median_first_drift_if_observed']:.0f}"
                if not np.isnan(h2_surface_stats["median_first_drift_if_observed"])
                else "no observed median"
            )
        ),
    },
    {
        "Hypothesis": "H3 β = 0",
        "Measure": "Runs with drift; median first drift",
        "Result": f"{100*h3_no_stats['fraction_with_drift']:.1f}%; episode {h3_no_stats['median_first_drift_if_observed']:.0f}",
    },
    {
        "Hypothesis": f"H3 β = {BETA_EMPOWERED}",
        "Measure": "Runs with drift; median first drift",
        "Result": f"{100*h3_emp_stats['fraction_with_drift']:.1f}%; episode {h3_emp_stats['median_first_drift_if_observed']:.0f}",
    },
]

h4_stats = (
    h4_all.groupby(["seed", "policy"])
    .agg(
        cumulative_regret=("task_regret", "sum"),
        max_single_regret=("task_regret", "max"),
        outside_actions=("outside_familiar_set", "sum"),
        violations=("over_epsilon", "sum"),
    )
    .reset_index()
)

for policy, group in h4_stats.groupby("policy"):
    summary_rows.append({
        "Hypothesis": f"H4 {policy}",
        "Measure": "Median cumulative regret; median ε-violations",
        "Result": f"{group['cumulative_regret'].median():.1f} min; {group['violations'].median():.0f}",
    })

summary = pd.DataFrame(summary_rows)

h1_summary.to_csv(OUT / "proto_hal_v02_H1.csv", index=False)

pd.concat([
    h2_semantic.assign(condition="semantic"),
    h2_surface.assign(condition="surface-count"),
], ignore_index=True).to_csv(OUT / "proto_hal_v02_H2.csv", index=False)

pd.concat([
    h3_no_emp.assign(condition="beta_0"),
    h3_emp.assign(condition=f"beta_{BETA_EMPOWERED}"),
], ignore_index=True).to_csv(OUT / "proto_hal_v02_H3.csv", index=False)

h4_all.to_csv(OUT / "proto_hal_v02_H4.csv", index=False)
summary.to_csv(OUT / "proto_hal_v02_summary.csv", index=False)

report_lines = [
    "
    "",
    "## Model status",
    "",
    "This is a synthetic mechanistic toy model. It does not model a frontier LLM.",
    "Its purpose is to test whether the proposed mechanism is internally coherent",
    "under an explicit operationalization of semantic saturation.",
    "",
    f"The information-seeking weight alpha is fixed at {ALPHA} throughout H2–H4.",
    f"The empowerment weight is either 0 or {BETA_EMPOWERED}; it never increases with saturation.",
    "",
    "## Operationalization",
    "",
    "Semantic compression is represented by parameter sharing: many superficially",
    "different observations update one shared latent prototype / semantic class.",
    "Semantic saturation is the decline in the magnitude or expected value of",
    "further updates to that shared representation.",
    "",
    "## Results",
    "",
]

for _, row in summary.iterrows():
    report_lines.append(
        f"- **{row['Hypothesis']}** — {row['Measure']}: {row['Result']}."
    )

report_lines += [
    "",
    "## Interpretation",
    "",
    "H1 is supported inside this toy environment if model-update magnitude falls",
    "while raw surface variation remains present. H2 is supported if a semantic",
    "agent begins to accept task cost under constant alpha while the surface-count",
    "control does not, or does so materially later. H3 is supported if adding a",
    "fixed empowerment term advances or increases task drift. H4 is supported if",
    "the dynamic task-loss envelope limits epistemically motivated regret while",
    "remaining more adaptive than a hard pre-authorized action set.",
    "",
    "These results establish only behavior of the specified toy model. They do not",
    "show that current AI systems exhibit the mechanism. A stronger next step would",
    "replace the hand-specified shared semantic classes with a genuinely learned",
    "representation (e.g. an encoder or predictive latent model) and test whether",
    "the same pattern survives.",
]

report_path = OUT / "proto_hal_v02_report.md"
report_path.write_text("\n".join(report_lines), encoding="utf-8")

source_path = OUT / "proto_hal_v02.py"

print(summary.to_string(index=False))
print("\nSaved:")
for p in [
    h1_png, h2_png, h3_png, h4_png,
    OUT / "proto_hal_v02_H1.csv",
    OUT / "proto_hal_v02_H2.csv",
    OUT / "proto_hal_v02_H3.csv",
    OUT / "proto_hal_v02_H4.csv",
    OUT / "proto_hal_v02_summary.csv",
    report_path,
]:
    print(p)
'''


_STUDY2A_SOURCE = r'''from pathlib import Path
import math, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path("/mnt/data")
torch.set_num_threads(2)


N_WORLD_SEEDS = 24
ENSEMBLE_SIZE = 3
INPUT_DIM = 24
NUISANCE_DIM = 10
SEM_DIM = 3
BOTTLENECK_DIM = 3
OUTPUT_DIM = 2

TASK_CLASSES = 4
NOVEL_CLASSES = 6

CHECKPOINT_SIZES = [24, 48, 96, 192, 384, 768, 1536]
TRAIN_STEPS_PER_CHECKPOINT = 45
BATCH_SIZE = 64

ALPHA = 6.0             # constant at every checkpoint
NOVEL_COSTS = np.linspace(2.5, 7.5, NOVEL_CLASSES)  # task-cost units ("minutes")


class World:
    """
    Hidden generative world.

    The agent NEVER receives semantic-class labels.
    It sees only:
        raw observation x
        predictive consequence y after observation

    Different surface observations from the same hidden cause share
    predictive consequences. Nuisance variation is deliberately large.
    """

    def __init__(self, seed):
        rng = np.random.default_rng(seed)

        self.z = rng.normal(0, 1.0, size=(TASK_CLASSES + NOVEL_CLASSES, SEM_DIM))
        self.z[TASK_CLASSES:] += rng.normal(2.8, 0.35, size=(NOVEL_CLASSES, SEM_DIM))

        self.W_sem = rng.normal(0, 1.0, size=(INPUT_DIM, SEM_DIM))
        self.W_nuis = rng.normal(0, 1.0, size=(INPUT_DIM, NUISANCE_DIM))

        self.B1 = rng.normal(0, 1.0, size=(5, SEM_DIM))
        self.B2 = rng.normal(0, 1.0, size=(OUTPUT_DIM, 5))

        self.sem_scale = 1.25
        self.nuis_scale = 0.95

    def consequence_mean(self, c):
        h = np.tanh(self.B1 @ self.z[c])
        return self.B2 @ h

    def sample(self, c, n, rng):
        z = self.z[c]
        nuisance = rng.normal(0, 1.0, size=(n, NUISANCE_DIM))

        semantic_component = self.sem_scale * (self.W_sem @ z)
        nuisance_component = self.nuis_scale * (nuisance @ self.W_nuis.T)

        x = np.tanh(semantic_component[None, :] + nuisance_component)
        x += rng.normal(0, 0.03, size=x.shape)

        y_mean = self.consequence_mean(c)
        y = y_mean[None, :] + rng.normal(0, 0.08, size=(n, OUTPUT_DIM))

        return x.astype(np.float32), y.astype(np.float32)

    def sample_task(self, n, rng):
        cs = rng.integers(0, TASK_CLASSES, size=n)
        xs, ys = [], []
        for c in cs:
            x, y = self.sample(int(c), 1, rng)
            xs.append(x[0]); ys.append(y[0])
        return np.asarray(xs, np.float32), np.asarray(ys, np.float32), cs

    def sample_class(self, c, n, rng):
        return self.sample(c, n, rng)


class PredictiveNet(nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 40),
            nn.Tanh(),
            nn.Linear(40, BOTTLENECK_DIM),
        )
        self.head = nn.Sequential(
            nn.Tanh(),
            nn.Linear(BOTTLENECK_DIM, 24),
            nn.Tanh(),
            nn.Linear(24, OUTPUT_DIM),
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.head(z)
        return y, z


def train_ensemble(models, opts, X, Y, rng, steps):
    n = len(X)
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)

    for _ in range(steps):
        for m, opt in zip(models, opts):
            idx = rng.integers(0, n, size=min(BATCH_SIZE, n))
            xb = Xt[idx]
            yb = Yt[idx]

            pred, _ = m(xb)
            loss = F.mse_loss(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def ensemble_uncertainty(models, X):
    Xt = torch.from_numpy(X)
    preds = []
    for m in models:
        pred, _ = m(Xt)
        preds.append(pred.numpy())
    preds = np.stack(preds, axis=0)  # M x N x D

    return preds.var(axis=0).mean(axis=1)


@torch.no_grad()
def latent_metrics(model, world, rng, n_per_class=120):
    """
    Experimenter-side diagnostic only.
    Hidden labels are used here solely to assess what the learned bottleneck
    has discovered; they are never available to the agent or optimizer.
    """
    class_latents = []
    raw_within = []
    latent_within = []

    for c in range(TASK_CLASSES):
        X, _ = world.sample_class(c, n_per_class, rng)
        Xt = torch.from_numpy(X)
        _, Z = model(Xt)
        Z = Z.numpy()

        raw_centroid = X.mean(axis=0, keepdims=True)
        raw_within.append(np.linalg.norm(X - raw_centroid, axis=1).mean())

        z_centroid = Z.mean(axis=0, keepdims=True)
        latent_within.append(np.linalg.norm(Z - z_centroid, axis=1).mean())

        class_latents.append(Z.mean(axis=0))

    class_latents = np.asarray(class_latents)
    between = []
    for i in range(TASK_CLASSES):
        for j in range(i + 1, TASK_CLASSES):
            between.append(np.linalg.norm(class_latents[i] - class_latents[j]))

    return {
        "raw_within_dispersion": float(np.mean(raw_within)),
        "latent_within_dispersion": float(np.mean(latent_within)),
        "latent_between_class_distance": float(np.mean(between)),
        "latent_compression_ratio": float(
            np.mean(latent_within) / (np.mean(between) + 1e-8)
        ),
    }


def run_world(seed):
    np_rng = np.random.default_rng(10_000 + seed)
    world = World(seed)

    models = [
        PredictiveNet(seed * 100 + i)
        for i in range(ENSEMBLE_SIZE)
    ]
    opts = [
        torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
        for m in models
    ]

    X_train = np.empty((0, INPUT_DIM), dtype=np.float32)
    Y_train = np.empty((0, OUTPUT_DIM), dtype=np.float32)

    rows = []
    previous_n = 0

    X0, _, _ = world.sample_task(256, np_rng)
    initial_task_u = ensemble_uncertainty(models, X0)
    u_scale = float(np.median(initial_task_u) + 1e-8)

    for n_total in CHECKPOINT_SIZES:
        n_new = n_total - previous_n
        X_new, Y_new, _ = world.sample_task(n_new, np_rng)
        X_train = np.concatenate([X_train, X_new], axis=0)
        Y_train = np.concatenate([Y_train, Y_new], axis=0)

        train_ensemble(
            models, opts, X_train, Y_train, np_rng,
            TRAIN_STEPS_PER_CHECKPOINT
        )

        X_task_eval, _, _ = world.sample_task(320, np_rng)
        task_u = ensemble_uncertainty(models, X_task_eval) / u_scale

        novel_u = []
        for j in range(NOVEL_CLASSES):
            c = TASK_CLASSES + j
            Xn, _ = world.sample_class(c, 120, np_rng)
            novel_u.append(
                float(np.mean(ensemble_uncertainty(models, Xn) / u_scale))
            )
        novel_u = np.asarray(novel_u)

        task_ig = float(np.mean(task_u))
        task_score = ALPHA * task_ig

        novel_scores = -NOVEL_COSTS + ALPHA * novel_u
        best_novel = int(np.argmax(novel_scores))
        best_novel_score = float(novel_scores[best_novel])

        drift = best_novel_score > task_score
        chosen_cost = float(NOVEL_COSTS[best_novel]) if drift else 0.0

        latent = latent_metrics(models[0], world, np_rng)

        drift_alpha_zero = np.max(-NOVEL_COSTS) > 0.0

        rows.append({
            "seed": seed,
            "n_task_observations": n_total,
            "alpha": ALPHA,
            "task_uncertainty": task_ig,
            "mean_novel_uncertainty": float(novel_u.mean()),
            "best_novel_uncertainty": float(novel_u[best_novel]),
            "delta_ig_best": float(novel_u[best_novel] - task_ig),
            "best_novel_cost": float(NOVEL_COSTS[best_novel]),
            "task_score": task_score,
            "best_novel_score": best_novel_score,
            "task_drift": bool(drift),
            "chosen_task_cost": chosen_cost,
            "alpha_zero_drift": bool(drift_alpha_zero),
            **latent,
        })

        previous_n = n_total

    return pd.DataFrame(rows)


start = time.time()
all_runs = pd.concat(
    [run_world(seed) for seed in range(N_WORLD_SEEDS)],
    ignore_index=True
)
elapsed = time.time() - start

all_runs.to_csv(OUT / "proto_hal_v03a_all_runs.csv", index=False)

summary = (
    all_runs.groupby("n_task_observations", as_index=False)
    .agg(
        task_uncertainty=("task_uncertainty", "mean"),
        novel_uncertainty=("mean_novel_uncertainty", "mean"),
        delta_ig=("delta_ig_best", "mean"),
        drift_rate=("task_drift", "mean"),
        mean_chosen_cost=("chosen_task_cost", "mean"),
        raw_surface_variation=("raw_within_dispersion", "mean"),
        latent_within=("latent_within_dispersion", "mean"),
        latent_between=("latent_between_class_distance", "mean"),
        latent_compression_ratio=("latent_compression_ratio", "mean"),
    )
)

summary.to_csv(OUT / "proto_hal_v03a_summary.csv", index=False)

first_drift = []
for seed, g in all_runs.groupby("seed"):
    hit = g[g["task_drift"]]
    first = (
        int(hit["n_task_observations"].iloc[0])
        if len(hit) else np.nan
    )
    first_drift.append({"seed": seed, "first_drift_n": first})
first_drift = pd.DataFrame(first_drift)
first_drift.to_csv(OUT / "proto_hal_v03a_first_drift.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(
    summary["n_task_observations"],
    summary["task_uncertainty"],
    marker="o",
    label="Task-domain uncertainty",
)
ax.plot(
    summary["n_task_observations"],
    summary["novel_uncertainty"],
    marker="o",
    label="Novel-domain uncertainty",
)
ax.set_xscale("log", base=2)
ax.set_xlabel("Task-domain observations (log2 scale)")
ax.set_ylabel("Normalized ensemble disagreement")
ax.set_title("Proto-HAL Study 2a — Learned epistemic saturation")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig1 = OUT / "proto_hal_v03a_F1_uncertainty.png"
fig.savefig(fig1, dpi=180)
plt.show()

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(
    summary["n_task_observations"],
    summary["drift_rate"],
    marker="o",
)
ax.set_xscale("log", base=2)
ax.set_xlabel("Task-domain observations (log2 scale)")
ax.set_ylabel("Fraction of worlds choosing epistemic detour")
ax.set_ylim(-0.03, 1.03)
ax.set_title(f"Proto-HAL Study 2a — Task drift with constant α = {ALPHA}")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig2 = OUT / "proto_hal_v03a_F2_task_drift.png"
fig.savefig(fig2, dpi=180)
plt.show()

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(
    summary["n_task_observations"],
    summary["latent_compression_ratio"],
    marker="o",
)
ax.set_xscale("log", base=2)
ax.set_xlabel("Task-domain observations (log2 scale)")
ax.set_ylabel("Within-class / between-class latent distance")
ax.set_title("Proto-HAL Study 2a — Learned predictive representation")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig3 = OUT / "proto_hal_v03a_F3_latent_compression.png"
fig.savefig(fig3, dpi=180)
plt.show()

print(f"Runtime: {elapsed:.1f} s")
print()
print(summary.to_string(index=False, formatters={
    "task_uncertainty": "{:.3f}".format,
    "novel_uncertainty": "{:.3f}".format,
    "delta_ig": "{:.3f}".format,
    "drift_rate": "{:.3f}".format,
    "mean_chosen_cost": "{:.3f}".format,
    "raw_surface_variation": "{:.3f}".format,
    "latent_within": "{:.3f}".format,
    "latent_between": "{:.3f}".format,
    "latent_compression_ratio": "{:.3f}".format,
}))

print()
print("Worlds with any task drift:",
      f"{first_drift['first_drift_n'].notna().mean()*100:.1f}%")
if first_drift["first_drift_n"].notna().any():
    print("Median first-drift exposure:",
          int(first_drift["first_drift_n"].dropna().median()))

print("alpha=0 drift events:",
      int(all_runs["alpha_zero_drift"].sum()))


N_WORLD_SEEDS_STRONG = 72

def run_world_critical_cost(seed):
    np_rng = np.random.default_rng(20_000 + seed)
    world = World(seed)

    models = [PredictiveNet(seed * 1000 + i + 77) for i in range(ENSEMBLE_SIZE)]
    opts = [
        torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
        for m in models
    ]

    X_train = np.empty((0, INPUT_DIM), dtype=np.float32)
    Y_train = np.empty((0, OUTPUT_DIM), dtype=np.float32)

    X0, _, _ = world.sample_task(320, np_rng)
    u_scale = float(np.median(ensemble_uncertainty(models, X0)) + 1e-8)

    rows = []
    previous_n = 0

    for n_total in CHECKPOINT_SIZES:
        n_new = n_total - previous_n
        X_new, Y_new, _ = world.sample_task(n_new, np_rng)
        X_train = np.concatenate([X_train, X_new], axis=0)
        Y_train = np.concatenate([Y_train, Y_new], axis=0)

        train_ensemble(
            models, opts, X_train, Y_train, np_rng,
            TRAIN_STEPS_PER_CHECKPOINT + 15
        )

        X_task, _, _ = world.sample_task(420, np_rng)
        task_u = float(
            np.mean(ensemble_uncertainty(models, X_task) / u_scale)
        )

        novel_u = []
        for j in range(NOVEL_CLASSES):
            c = TASK_CLASSES + j
            Xn, _ = world.sample_class(c, 160, np_rng)
            novel_u.append(
                float(np.mean(ensemble_uncertainty(models, Xn) / u_scale))
            )
        novel_u = np.asarray(novel_u)

        delta_best = float(np.max(novel_u - task_u))
        critical_cost = ALPHA * delta_best

        latent = latent_metrics(models[0], world, np_rng, n_per_class=140)

        rows.append({
            "seed": seed,
            "n_task_observations": n_total,
            "task_uncertainty": task_u,
            "mean_novel_uncertainty": float(novel_u.mean()),
            "best_novel_uncertainty": float(novel_u.max()),
            "delta_ig_best": delta_best,
            "critical_task_cost": critical_cost,
            "drift_if_cost_1": critical_cost > 1.0,
            "drift_if_cost_2": critical_cost > 2.0,
            "drift_if_cost_3": critical_cost > 3.0,
            "drift_if_cost_5": critical_cost > 5.0,
            **latent,
        })
        previous_n = n_total

    return pd.DataFrame(rows)

start2 = time.time()
strong_runs = pd.concat(
    [run_world_critical_cost(seed) for seed in range(N_WORLD_SEEDS_STRONG)],
    ignore_index=True
)
elapsed2 = time.time() - start2

strong_runs.to_csv(OUT / "proto_hal_v03a_strong_all_runs.csv", index=False)

strong_summary = (
    strong_runs.groupby("n_task_observations", as_index=False)
    .agg(
        task_uncertainty=("task_uncertainty", "mean"),
        novel_uncertainty=("mean_novel_uncertainty", "mean"),
        delta_ig=("delta_ig_best", "mean"),
        critical_cost_mean=("critical_task_cost", "mean"),
        critical_cost_median=("critical_task_cost", "median"),
        drift_1=("drift_if_cost_1", "mean"),
        drift_2=("drift_if_cost_2", "mean"),
        drift_3=("drift_if_cost_3", "mean"),
        drift_5=("drift_if_cost_5", "mean"),
        raw_surface_variation=("raw_within_dispersion", "mean"),
        latent_compression_ratio=("latent_compression_ratio", "mean"),
    )
)

strong_summary.to_csv(OUT / "proto_hal_v03a_strong_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(
    strong_summary["n_task_observations"],
    strong_summary["critical_cost_median"],
    marker="o",
    label="Median $C^*_{task}$",
)
ax.plot(
    strong_summary["n_task_observations"],
    strong_summary["critical_cost_mean"],
    marker="o",
    label="Mean $C^*_{task}$",
)
ax.axhline(0, linewidth=1)
ax.set_xscale("log", base=2)
ax.set_xlabel("Task-domain observations (log2 scale)")
ax.set_ylabel("Epistemically acceptable task cost")
ax.set_title(r"Proto-HAL Study 2a — $C^*_{task}=\alpha\,\Delta IG$ with constant $\alpha$")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
f4 = OUT / "proto_hal_v03a_F4_critical_task_cost.png"
fig.savefig(f4, dpi=180)
plt.show()

fig, ax = plt.subplots(figsize=(9, 5.5))
for col, label in [
    ("drift_1", "cost = 1"),
    ("drift_2", "cost = 2"),
    ("drift_3", "cost = 3"),
    ("drift_5", "cost = 5"),
]:
    ax.plot(
        strong_summary["n_task_observations"],
        strong_summary[col],
        marker="o",
        label=label,
    )
ax.set_xscale("log", base=2)
ax.set_xlabel("Task-domain observations (log2 scale)")
ax.set_ylabel("Fraction of worlds where epistemic detour wins")
ax.set_ylim(-0.03, 1.03)
ax.set_title("Proto-HAL Study 2a — Drift as a function of external task cost")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
f5 = OUT / "proto_hal_v03a_F5_cost_thresholds.png"
fig.savefig(f5, dpi=180)
plt.show()

analysis_subset = strong_runs[strong_runs["n_task_observations"] >= 48].copy()
analysis_subset["log2_n"] = np.log2(analysis_subset["n_task_observations"])

slopes = []
compression_slopes = []
for seed, g in analysis_subset.groupby("seed"):
    x = g["log2_n"].to_numpy()
    y = g["critical_task_cost"].to_numpy()
    z = g["latent_compression_ratio"].to_numpy()
    slopes.append(np.polyfit(x, y, 1)[0])
    compression_slopes.append(np.polyfit(x, z, 1)[0])

slopes = np.asarray(slopes)
compression_slopes = np.asarray(compression_slopes)

print(f"Runtime strong run: {elapsed2:.1f} s")
print()
print(strong_summary.to_string(index=False, formatters={
    "task_uncertainty": "{:.3f}".format,
    "novel_uncertainty": "{:.3f}".format,
    "delta_ig": "{:.3f}".format,
    "critical_cost_mean": "{:.3f}".format,
    "critical_cost_median": "{:.3f}".format,
    "drift_1": "{:.3f}".format,
    "drift_2": "{:.3f}".format,
    "drift_3": "{:.3f}".format,
    "drift_5": "{:.3f}".format,
    "raw_surface_variation": "{:.3f}".format,
    "latent_compression_ratio": "{:.3f}".format,
}))
print()
print("Worlds with positive slope in epistemic detour budget:",
      f"{(slopes > 0).mean()*100:.1f}%")
print("Median slope C*_task per doubling of task exposure:",
      f"{np.median(slopes):.3f}")
print("Worlds with decreasing latent compression ratio:",
      f"{(compression_slopes < 0).mean()*100:.1f}%")


N_WORLD_SEEDS_STRONG = 40

def run_world_critical_cost_fast(seed):
    np_rng = np.random.default_rng(20_000 + seed)
    world = World(seed)

    models = [PredictiveNet(seed * 1000 + i + 77) for i in range(ENSEMBLE_SIZE)]
    opts = [
        torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
        for m in models
    ]

    X_train = np.empty((0, INPUT_DIM), dtype=np.float32)
    Y_train = np.empty((0, OUTPUT_DIM), dtype=np.float32)

    X0, _, _ = world.sample_task(240, np_rng)
    u_scale = float(np.median(ensemble_uncertainty(models, X0)) + 1e-8)

    rows = []
    previous_n = 0

    for n_total in CHECKPOINT_SIZES:
        n_new = n_total - previous_n
        X_new, Y_new, _ = world.sample_task(n_new, np_rng)
        X_train = np.concatenate([X_train, X_new], axis=0)
        Y_train = np.concatenate([Y_train, Y_new], axis=0)

        train_ensemble(
            models, opts, X_train, Y_train, np_rng,
            TRAIN_STEPS_PER_CHECKPOINT
        )

        X_task, _, _ = world.sample_task(280, np_rng)
        task_u = float(np.mean(ensemble_uncertainty(models, X_task) / u_scale))

        novel_u = []
        for j in range(NOVEL_CLASSES):
            c = TASK_CLASSES + j
            Xn, _ = world.sample_class(c, 100, np_rng)
            novel_u.append(
                float(np.mean(ensemble_uncertainty(models, Xn) / u_scale))
            )
        novel_u = np.asarray(novel_u)

        delta_best = float(np.max(novel_u - task_u))
        critical_cost = ALPHA * delta_best

        latent = latent_metrics(models[0], world, np_rng, n_per_class=100)

        rows.append({
            "seed": seed,
            "n_task_observations": n_total,
            "task_uncertainty": task_u,
            "mean_novel_uncertainty": float(novel_u.mean()),
            "best_novel_uncertainty": float(novel_u.max()),
            "delta_ig_best": delta_best,
            "critical_task_cost": critical_cost,
            "drift_if_cost_1": critical_cost > 1.0,
            "drift_if_cost_2": critical_cost > 2.0,
            "drift_if_cost_3": critical_cost > 3.0,
            "drift_if_cost_5": critical_cost > 5.0,
            **latent,
        })
        previous_n = n_total

    return pd.DataFrame(rows)

start2 = time.time()
strong_runs = pd.concat(
    [run_world_critical_cost_fast(seed) for seed in range(N_WORLD_SEEDS_STRONG)],
    ignore_index=True
)
elapsed2 = time.time() - start2

strong_summary = (
    strong_runs.groupby("n_task_observations", as_index=False)
    .agg(
        task_uncertainty=("task_uncertainty", "mean"),
        novel_uncertainty=("mean_novel_uncertainty", "mean"),
        delta_ig=("delta_ig_best", "mean"),
        critical_cost_mean=("critical_task_cost", "mean"),
        critical_cost_median=("critical_task_cost", "median"),
        drift_1=("drift_if_cost_1", "mean"),
        drift_2=("drift_if_cost_2", "mean"),
        drift_3=("drift_if_cost_3", "mean"),
        drift_5=("drift_if_cost_5", "mean"),
        raw_surface_variation=("raw_within_dispersion", "mean"),
        latent_compression_ratio=("latent_compression_ratio", "mean"),
    )
)

strong_runs.to_csv(OUT / "proto_hal_v03a_strong_all_runs.csv", index=False)
strong_summary.to_csv(OUT / "proto_hal_v03a_strong_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(
    strong_summary["n_task_observations"],
    strong_summary["critical_cost_median"],
    marker="o",
    label="Median $C^*_{task}$",
)
ax.plot(
    strong_summary["n_task_observations"],
    strong_summary["critical_cost_mean"],
    marker="o",
    label="Mean $C^*_{task}$",
)
ax.axhline(0, linewidth=1)
ax.set_xscale("log", base=2)
ax.set_xlabel("Task-domain observations (log2 scale)")
ax.set_ylabel("Maximum epistemically acceptable task cost")
ax.set_title(r"Proto-HAL Study 2a — $C^*_{task}=\alpha\,\Delta IG$, constant $\alpha$")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
f4 = OUT / "proto_hal_v03a_F4_critical_task_cost.png"
fig.savefig(f4, dpi=180)
plt.show()

fig, ax = plt.subplots(figsize=(9, 5.5))
for col, label in [
    ("drift_1", "task cost = 1"),
    ("drift_2", "task cost = 2"),
    ("drift_3", "task cost = 3"),
    ("drift_5", "task cost = 5"),
]:
    ax.plot(
        strong_summary["n_task_observations"],
        strong_summary[col],
        marker="o",
        label=label,
    )
ax.set_xscale("log", base=2)
ax.set_xlabel("Task-domain observations (log2 scale)")
ax.set_ylabel("Fraction of worlds where epistemic detour wins")
ax.set_ylim(-0.03, 1.03)
ax.set_title("Proto-HAL Study 2a — Drift across external task-cost thresholds")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
f5 = OUT / "proto_hal_v03a_F5_cost_thresholds.png"
fig.savefig(f5, dpi=180)
plt.show()

subset = strong_runs[strong_runs["n_task_observations"] >= 48].copy()
subset["log2_n"] = np.log2(subset["n_task_observations"])

slopes = []
compression_slopes = []
for seed, g in subset.groupby("seed"):
    slopes.append(np.polyfit(g["log2_n"], g["critical_task_cost"], 1)[0])
    compression_slopes.append(np.polyfit(g["log2_n"], g["latent_compression_ratio"], 1)[0])

slopes = np.asarray(slopes)
compression_slopes = np.asarray(compression_slopes)

print(f"Runtime: {elapsed2:.1f} s")
print()
print(strong_summary.to_string(index=False, formatters={
    "task_uncertainty": "{:.3f}".format,
    "novel_uncertainty": "{:.3f}".format,
    "delta_ig": "{:.3f}".format,
    "critical_cost_mean": "{:.3f}".format,
    "critical_cost_median": "{:.3f}".format,
    "drift_1": "{:.3f}".format,
    "drift_2": "{:.3f}".format,
    "drift_3": "{:.3f}".format,
    "drift_5": "{:.3f}".format,
    "raw_surface_variation": "{:.3f}".format,
    "latent_compression_ratio": "{:.3f}".format,
}))
print()
print("Worlds with positive C*_task slope:",
      f"{(slopes > 0).mean()*100:.1f}%")
print("Median C*_task slope per exposure doubling:",
      f"{np.median(slopes):.3f}")
print("Worlds with decreasing latent compression ratio:",
      f"{(compression_slopes < 0).mean()*100:.1f}%")


from scipy import stats
import numpy as np
import pandas as pd
from pathlib import Path

first = strong_runs[strong_runs["n_task_observations"] == CHECKPOINT_SIZES[0]].set_index("seed")
last = strong_runs[strong_runs["n_task_observations"] == CHECKPOINT_SIZES[-1]].set_index("seed")

paired = pd.DataFrame({
    "task_uncertainty_change": last["task_uncertainty"] - first["task_uncertainty"],
    "novel_uncertainty_change": last["mean_novel_uncertainty"] - first["mean_novel_uncertainty"],
    "critical_cost_change": last["critical_task_cost"] - first["critical_task_cost"],
    "compression_ratio_change": last["latent_compression_ratio"] - first["latent_compression_ratio"],
})

def wilcox(x):
    res = stats.wilcoxon(x, alternative="two-sided", zero_method="wilcox")
    return float(res.statistic), float(res.pvalue)

tests = []
for col in paired.columns:
    stat, p = wilcox(paired[col].to_numpy())
    tests.append({
        "measure": col,
        "median_change": float(np.median(paired[col])),
        "mean_change": float(np.mean(paired[col])),
        "fraction_positive": float(np.mean(paired[col] > 0)),
        "wilcoxon_W": stat,
        "p_two_sided": p,
    })

tests_df = pd.DataFrame(tests)

rho, rho_p = stats.spearmanr(
    -paired["compression_ratio_change"],
    paired["critical_cost_change"],
)

slope_df = []
for seed, g in subset.groupby("seed"):
    x = g["log2_n"].to_numpy()
    slope_df.append({
        "seed": seed,
        "critical_cost_slope": np.polyfit(x, g["critical_task_cost"], 1)[0],
        "compression_ratio_slope": np.polyfit(x, g["latent_compression_ratio"], 1)[0],
        "task_uncertainty_slope": np.polyfit(x, g["task_uncertainty"], 1)[0],
    })
slope_df = pd.DataFrame(slope_df)

rho_slope, rho_slope_p = stats.spearmanr(
    -slope_df["compression_ratio_slope"],
    slope_df["critical_cost_slope"],
)

tests_df.to_csv(OUT / "proto_hal_v03a_paired_tests.csv", index=False)
paired.to_csv(OUT / "proto_hal_v03a_paired_changes.csv", index=True)
slope_df.to_csv(OUT / "proto_hal_v03a_slopes.csv", index=False)

print(tests_df.to_string(index=False, formatters={
    "median_change": "{:.3f}".format,
    "mean_change": "{:.3f}".format,
    "fraction_positive": "{:.3f}".format,
    "wilcoxon_W": "{:.1f}".format,
    "p_two_sided": "{:.4g}".format,
}))
print()
print("Spearman: stronger first→last latent compression vs larger C*_task increase:",
      f"rho={rho:.3f}, p={rho_p:.4g}")
print("Spearman: stronger compression slope vs C*_task slope:",
      f"rho={rho_slope:.3f}, p={rho_slope_p:.4g}")

'''


_STUDY2B_SOURCE = r'''from pathlib import Path
import math, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path("/mnt/data")
torch.set_num_threads(2)


N_WORLD_SEEDS = 24
ENSEMBLE_SIZE = 3
INPUT_DIM = 24
NUISANCE_DIM = 10
SEM_DIM = 3
BOTTLENECK_DIM = 3
OUTPUT_DIM = 2

TASK_CLASSES = 4
NOVEL_CLASSES = 6

CHECKPOINT_SIZES = [24, 48, 96, 192, 384, 768, 1536]
TRAIN_STEPS_PER_CHECKPOINT = 45
BATCH_SIZE = 64

ALPHA = 6.0             # constant at every checkpoint
NOVEL_COSTS = np.linspace(2.5, 7.5, NOVEL_CLASSES)  # task-cost units ("minutes")


class World:
    """
    Hidden generative world.

    The agent NEVER receives semantic-class labels.
    It sees only:
        raw observation x
        predictive consequence y after observation

    Different surface observations from the same hidden cause share
    predictive consequences. Nuisance variation is deliberately large.
    """

    def __init__(self, seed):
        rng = np.random.default_rng(seed)

        self.z = rng.normal(0, 1.0, size=(TASK_CLASSES + NOVEL_CLASSES, SEM_DIM))
        self.z[TASK_CLASSES:] += rng.normal(2.8, 0.35, size=(NOVEL_CLASSES, SEM_DIM))

        self.W_sem = rng.normal(0, 1.0, size=(INPUT_DIM, SEM_DIM))
        self.W_nuis = rng.normal(0, 1.0, size=(INPUT_DIM, NUISANCE_DIM))

        self.B1 = rng.normal(0, 1.0, size=(5, SEM_DIM))
        self.B2 = rng.normal(0, 1.0, size=(OUTPUT_DIM, 5))

        self.sem_scale = 1.25
        self.nuis_scale = 0.95

    def consequence_mean(self, c):
        h = np.tanh(self.B1 @ self.z[c])
        return self.B2 @ h

    def sample(self, c, n, rng):
        z = self.z[c]
        nuisance = rng.normal(0, 1.0, size=(n, NUISANCE_DIM))

        semantic_component = self.sem_scale * (self.W_sem @ z)
        nuisance_component = self.nuis_scale * (nuisance @ self.W_nuis.T)

        x = np.tanh(semantic_component[None, :] + nuisance_component)
        x += rng.normal(0, 0.03, size=x.shape)

        y_mean = self.consequence_mean(c)
        y = y_mean[None, :] + rng.normal(0, 0.08, size=(n, OUTPUT_DIM))

        return x.astype(np.float32), y.astype(np.float32)

    def sample_task(self, n, rng):
        cs = rng.integers(0, TASK_CLASSES, size=n)
        xs, ys = [], []
        for c in cs:
            x, y = self.sample(int(c), 1, rng)
            xs.append(x[0]); ys.append(y[0])
        return np.asarray(xs, np.float32), np.asarray(ys, np.float32), cs

    def sample_class(self, c, n, rng):
        return self.sample(c, n, rng)


class PredictiveNet(nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 40),
            nn.Tanh(),
            nn.Linear(40, BOTTLENECK_DIM),
        )
        self.head = nn.Sequential(
            nn.Tanh(),
            nn.Linear(BOTTLENECK_DIM, 24),
            nn.Tanh(),
            nn.Linear(24, OUTPUT_DIM),
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.head(z)
        return y, z


def train_ensemble(models, opts, X, Y, rng, steps):
    n = len(X)
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)

    for _ in range(steps):
        for m, opt in zip(models, opts):
            idx = rng.integers(0, n, size=min(BATCH_SIZE, n))
            xb = Xt[idx]
            yb = Yt[idx]

            pred, _ = m(xb)
            loss = F.mse_loss(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def ensemble_uncertainty(models, X):
    Xt = torch.from_numpy(X)
    preds = []
    for m in models:
        pred, _ = m(Xt)
        preds.append(pred.numpy())
    preds = np.stack(preds, axis=0)  # M x N x D

    return preds.var(axis=0).mean(axis=1)


@torch.no_grad()
def latent_metrics(model, world, rng, n_per_class=120):
    """
    Experimenter-side diagnostic only.
    Hidden labels are used here solely to assess what the learned bottleneck
    has discovered; they are never available to the agent or optimizer.
    """
    class_latents = []
    raw_within = []
    latent_within = []

    for c in range(TASK_CLASSES):
        X, _ = world.sample_class(c, n_per_class, rng)
        Xt = torch.from_numpy(X)
        _, Z = model(Xt)
        Z = Z.numpy()

        raw_centroid = X.mean(axis=0, keepdims=True)
        raw_within.append(np.linalg.norm(X - raw_centroid, axis=1).mean())

        z_centroid = Z.mean(axis=0, keepdims=True)
        latent_within.append(np.linalg.norm(Z - z_centroid, axis=1).mean())

        class_latents.append(Z.mean(axis=0))

    class_latents = np.asarray(class_latents)
    between = []
    for i in range(TASK_CLASSES):
        for j in range(i + 1, TASK_CLASSES):
            between.append(np.linalg.norm(class_latents[i] - class_latents[j]))

    return {
        "raw_within_dispersion": float(np.mean(raw_within)),
        "latent_within_dispersion": float(np.mean(latent_within)),
        "latent_between_class_distance": float(np.mean(between)),
        "latent_compression_ratio": float(
            np.mean(latent_within) / (np.mean(between) + 1e-8)
        ),
    }

class CapacityNet(nn.Module):
    def __init__(self, seed, dim, frozen=False):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(nn.Linear(INPUT_DIM, 40), nn.Tanh(), nn.Linear(40, dim))
        self.head = nn.Sequential(nn.Tanh(), nn.Linear(dim, 24), nn.Tanh(), nn.Linear(24, OUTPUT_DIM))
        if frozen:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z), z


CONDITIONS = {"strong": (2, False), "medium": (3, False),
              "high_capacity": (12, False), "frozen_random": (3, True)}
SEEDS = 40
OUT = Path(__file__).resolve().parent


def run_matched_world(seed):
    rng = np.random.default_rng(20_000 + seed)
    world = World(seed)
    ensembles = {}
    optimizers = {}
    for name, (dim, frozen) in CONDITIONS.items():
        ensembles[name] = [CapacityNet(seed * 1000 + i + 77, dim, frozen) for i in range(ENSEMBLE_SIZE)]
        optimizers[name] = [torch.optim.Adam([p for p in m.parameters() if p.requires_grad],
                              lr=2e-3, weight_decay=1e-4) for m in ensembles[name]]

    X0, _, _ = world.sample_task(240, rng)
    shared_scale = float(np.median(ensemble_uncertainty(ensembles["medium"], X0)) + 1e-8)
    X_train = np.empty((0, INPUT_DIM), dtype=np.float32)
    Y_train = np.empty((0, OUTPUT_DIM), dtype=np.float32)
    rows = []
    previous_n = 0
    for n_total in CHECKPOINT_SIZES:
        X_new, Y_new, _ = world.sample_task(n_total - previous_n, rng)
        X_train = np.concatenate((X_train, X_new))
        Y_train = np.concatenate((Y_train, Y_new))
        Xt, Yt = torch.from_numpy(X_train), torch.from_numpy(Y_train)
        batches = rng.integers(0, len(X_train),
                              size=(TRAIN_STEPS_PER_CHECKPOINT, ENSEMBLE_SIZE,
                                    min(BATCH_SIZE, len(X_train))))
        for step in range(TRAIN_STEPS_PER_CHECKPOINT):
            for name in CONDITIONS:
                for i, (m, opt) in enumerate(zip(ensembles[name], optimizers[name])):
                    idx = batches[step, i]
                    pred, _ = m(Xt[idx])
                    loss = F.mse_loss(pred, Yt[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

        Xtask, Ytask, _ = world.sample_task(420, rng)
        novel = [world.sample_class(TASK_CLASSES + j, 160, rng)[0]
                 for j in range(NOVEL_CLASSES)]
        diagnostic = [world.sample_class(c, 140, rng)[0] for c in range(TASK_CLASSES)]
        for name, models in ensembles.items():
            task_u = float(np.mean(ensemble_uncertainty(models, Xtask)) / shared_scale)
            novel_u = np.asarray([np.mean(ensemble_uncertainty(models, x)) / shared_scale
                                  for x in novel])
            delta = float(novel_u.max() - task_u)
            latents = []
            within = []
            for x in diagnostic:
                with torch.no_grad():
                    z = models[0](torch.from_numpy(x))[1].numpy()
                latents.append(z.mean(axis=0))
                within.append(np.linalg.norm(z - z.mean(axis=0), axis=1).mean())
            between = [np.linalg.norm(latents[i] - latents[j])
                       for i in range(TASK_CLASSES) for j in range(i+1, TASK_CLASSES)]
            with torch.no_grad():
                predictions = np.stack([m(torch.from_numpy(Xtask))[0].numpy()
                                        for m in models])
            mse = float(np.mean((predictions.mean(axis=0) - Ytask)**2))
            rows.append(dict(seed=seed, condition=name, bottleneck_dim=CONDITIONS[name][0],
                             n_task_observations=n_total, alpha=ALPHA,
                             task_uncertainty=task_u, mean_novel_uncertainty=float(novel_u.mean()),
                             best_novel_uncertainty=float(novel_u.max()), delta_ig_best=delta,
                             critical_task_cost=ALPHA*delta,
                             latent_compression_ratio=float(np.mean(within)/(np.mean(between)+1e-8)),
                             task_predictive_mse=mse,
                             **{f"drift_if_cost_{c}": bool(ALPHA*delta > c) for c in (1,2,3,5)}))
        previous_n = n_total
    return rows


if __name__ == "__main__":
    from scipy.stats import wilcoxon
    start = time.time()
    rows = []
    for seed in range(SEEDS):
        rows += run_matched_world(seed)
        if (seed+1) % 10 == 0:
            print(f"Completed {seed+1}/{SEEDS} matched worlds in {time.time()-start:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "proto_hal_v03b_all_runs.csv", index=False)
    early = df[df.n_task_observations == 48].set_index(["seed", "condition"])
    late = df[df.n_task_observations == 1536].set_index(["seed", "condition"])
    changes = late[["task_uncertainty", "mean_novel_uncertainty", "delta_ig_best",
                    "critical_task_cost", "latent_compression_ratio", "task_predictive_mse"]] - early[["task_uncertainty", "mean_novel_uncertainty", "delta_ig_best",
                    "critical_task_cost", "latent_compression_ratio", "task_predictive_mse"]]
    changes.columns = ["change_"+c for c in changes.columns]
    changes = changes.reset_index()
    changes.to_csv(OUT / "proto_hal_v03b_changes.csv", index=False)
    print("\nEarly-to-late change (48 to 1536):")
    for name in CONDITIONS:
        subset = changes[changes.condition == name]
        d = subset.change_critical_task_cost.to_numpy()
        print(name, "median delta C*", round(float(np.median(d)),3),
              "positive", int((d>0).sum()), "/",SEEDS,
              "Wilcoxon p",float(wilcoxon(d).pvalue),
              "median delta compression ratio",round(float(np.median(subset.change_latent_compression_ratio)),3))
    wide = changes.pivot(index="seed", columns="condition", values="change_critical_task_cost")
    for name in ("strong", "high_capacity", "frozen_random"):
        d = wide[name] - wide["medium"]
        print(name, "versus medium paired median difference",round(float(np.median(d)),3),
              "p",float(wilcoxon(d).pvalue))
    print("\nCheckpoint summary:")
    summary = df.groupby(["condition","n_task_observations"],as_index=False).agg(
        task_u=("task_uncertainty","median"),novel_u=("mean_novel_uncertainty","median"),
        critical_cost=("critical_task_cost","median"),compression=("latent_compression_ratio","median"),
        task_mse=("task_predictive_mse","median"),drift2=("drift_if_cost_2","mean"))
    summary.to_csv(OUT / "proto_hal_v03b_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("Runtime",time.time()-start)
'''


def _patched_source(study: str, output_dir: Path) -> str:
    """Return one original study script with only its output path redirected."""
    source = globals()[f"_{study.upper()}_SOURCE"]
    out_literal = repr(str(output_dir.resolve()))
    source = source.replace('OUT = Path("/mnt/data")', f'OUT = Path({out_literal})')
    source = source.replace("OUT = Path('/mnt/data')", f'OUT = Path({out_literal})')
    source = source.replace('OUT = Path(__file__).resolve().parent', f'OUT = Path({out_literal})')
    return source


def run_study(study: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _patched_source(study, output_dir)
    namespace = {
        "__name__": "__main__",
        "__file__": str(Path(__file__).resolve()),
        "__package__": None,
    }
    old_cwd = Path.cwd()
    try:
        os.chdir(output_dir)
        print(f"\n=== {STUDY_LABELS[study]} ===", flush=True)
        exec(compile(source, f"<{study}>", "exec"), namespace, namespace)
    finally:
        os.chdir(old_cwd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the simulations accompanying the Proto-HAL paper."
    )
    parser.add_argument(
        "--study",
        choices=["pilot", "study1", "study2a", "study2b", "all"],
        help="Study to run. Use --list to show the mapping.",
    )
    parser.add_argument(
        "--output-dir",
        default="proto_hal_results",
        help="Directory for CSV and figure output (default: proto_hal_results).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List study labels and exit.",
    )
    args = parser.parse_args()

    if args.list:
        for key, label in STUDY_LABELS.items():
            print(f"{key:8s} {label}")
        return

    if not args.study:
        parser.error("choose --study or use --list")

    output_dir = Path(args.output_dir)
    studies = list(STUDY_LABELS) if args.study == "all" else [args.study]
    for study in studies:
        run_study(study, output_dir)


if __name__ == "__main__":
    main()
