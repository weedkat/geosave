"""Plot current memory through the single-process lifecycle experiment."""

import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path(__file__).parent
rows = json.loads((folder / "results.json").read_text())["results"]
phases = ["before", "prepared", "epoch_1", "epoch_2", "epoch_3", "closed", "collected"]
labels = ["Start", "Prepared", "Epoch 1", "Epoch 2", "Epoch 3", "Closed", "After GC"]
upper = 0
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, constrained_layout=True)
for ax, layout, title in zip(
    axes,
    ["cache", "saved"],
    ["Temporary source cache", "Persistent final chips"],
    strict=True,
):
    for policy, label in [
        ("retain", "Retain mappings"),
        ("release", "Release by block / batch"),
    ]:
        runs = [
            row
            for row in rows
            if row["size"] == 4096
            and row["layout"] == layout
            and row["policy"] == policy
        ]
        series = []
        for phase in phases:
            series.append(
                median(
                    [
                        row["points"][phase]["rss_mib"]
                        - row["points"]["before"]["rss_mib"]
                        for row in runs
                    ]
                )
            )
        upper = max(upper, max(series))
        ax.plot(labels, series, marker="o", linewidth=2, label=label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper left")
axes[0].set_ylim(0, upper * 1.1)
axes[0].set_ylabel("Current RSS above start (MiB)")
fig.suptitle(
    "Memory through preparation and reading\n4096 × 4109, three bands · median of three runs · zero DataLoader workers"
)
fig.savefig(folder / "memory-lifecycle.png", dpi=160)
