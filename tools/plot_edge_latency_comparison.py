"""Create a publication-ready embedded HAR latency comparison.

Literature values are transcribed from:
  [1] Wang et al., Real-Time Human Action Recognition on Embedded Platforms,
      arXiv:2409.05662, Table III (end-to-end, Jetson Xavier NX).
  [2] Lin et al., TSM: Temporal Shift Module for Efficient Video Understanding,
      ICCV 2019, Table 8 (model inference, Jetson Nano GPU).
  [3] Efficient Human Action Recognition with Mixed-Precision V-JEPA on
      Embedded Devices, ICICT 2026, Table 6 (model inference, AGX Orin).

The two VPOCLIP measurements are supplied by the project owner. End-to-end and
model-only measurements intentionally occupy separate panels because their
timing boundaries are not interchangeable.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figures"


def add_labels(ax, bars, values, suffix=" ms"):
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:g}{suffix}",
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold" if bar.get_facecolor()[:3] == (0.12156862745098039, 0.4666666666666667, 0.7058823529411765) else "normal",
        )


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (ax_e2e, ax_core) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.4),
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.58},
    )
    fig.patch.set_facecolor("white")

    # The RT-HARE paper reports all five literature E2E entries on the same
    # Xavier NX testbed. VPOCLIP was measured on the AGX Xavier deployment.
    e2e_names = [
        "RGB-only\nXavier NX",
        "RT-HARE + DLA\nXavier NX",
        "RT-HARE\nXavier NX",
        "VPOCLIP (ours)\nJetson AGX Xavier",
        "RAFT two-stream\nXavier NX",
        "TV-L1 two-stream\nXavier NX",
    ]
    e2e_values = [24.52, 62.98, 68.83, 130.0, 169.74, 614.01]
    e2e_colors = ["#9AA0A6", "#2A9D8F", "#55A868", "#1F77B4", "#D08C32", "#C95D3F"]

    bars = ax_e2e.barh(e2e_names, e2e_values, color=e2e_colors, height=0.68)
    ax_e2e.set_xscale("log")
    ax_e2e.set_xlim(18, 900)
    ax_e2e.set_xticks([20, 50, 100, 200, 500])
    ax_e2e.xaxis.set_major_formatter(ScalarFormatter())
    ax_e2e.axvline(200, color="#7A1F1F", linestyle="--", linewidth=1.2)
    ax_e2e.text(
        200,
        0.985,
        "200 ms real-time decision deadline",
        transform=ax_e2e.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
        color="#7A1F1F",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
    )
    add_labels(ax_e2e, bars, e2e_values)
    ax_e2e.invert_yaxis()
    ax_e2e.set_title("(a) Embedded end-to-end action-recognition pipelines", loc="left", fontweight="bold")
    ax_e2e.set_xlabel("End-to-end latency per decision (ms, logarithmic scale; lower is better)")
    ax_e2e.grid(axis="x", which="major", color="#D9D9D9", linewidth=0.8)
    ax_e2e.set_axisbelow(True)

    core_names = [
        "V-JEPA INT8/FP32\nJetson AGX Orin",
        "Online TSM\nJetson Nano GPU",
        "VPOCLIP core (ours)\nJetson AGX Xavier",
    ]
    core_values = [13.0, 13.4, 30.0]
    core_colors = ["#8172B2", "#CCB974", "#1F77B4"]

    bars = ax_core.barh(core_names, core_values, color=core_colors, height=0.62)
    add_labels(ax_core, bars, core_values)
    ax_core.invert_yaxis()
    ax_core.set_xlim(0, 40)
    ax_core.set_xticks([0, 10, 20, 30, 40])
    ax_core.set_title("(b) Model/core inference only (not end-to-end)", loc="left", fontweight="bold")
    ax_core.set_xlabel("Core/model latency (ms; lower is better)")
    ax_core.grid(axis="x", color="#D9D9D9", linewidth=0.8)
    ax_core.set_axisbelow(True)

    for ax in (ax_e2e, ax_core):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    fig.suptitle(
        "On-device latency comparison for human action recognition",
        x=0.09,
        y=0.985,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.09,
        0.014,
        "Protocols, datasets, precision, input windows, and Jetson variants differ. E2E includes preprocessing/post-processing where\n"
        "reported; core/model values exclude the surrounding perception pipeline.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#3F3F3F",
    )
    fig.subplots_adjust(left=0.31, right=0.94, top=0.91, bottom=0.13)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "edge_latency_comparison.png", dpi=320, facecolor="white")
    fig.savefig(OUTPUT_DIR / "edge_latency_comparison.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
