"""Operational outcomes plot for the weight-loss randomized trial (Fig. 2c).

The input JSON is produced by ``clinical_trial/checkin_analysis/
enhanced_feedback_mediation.py``. The parser also accepts the legacy manuscript
analysis JSON keys used by the original plotting script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ARM_ORDER = ("Human", "EPS-human")
ARM_COLORS = {"Human": "#AAB7C0", "EPS-human": "#0072B2"}
ARM_EDGES = {"Human": "#6E7C86", "EPS-human": "#004C79"}
REVIEW_COLORS = {"Accepted unchanged": "#8DC6E8", "Edited before delivery": "#D95F02"}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_quality(summary: Dict[str, Any]) -> Dict[str, Any]:
    for key in (
        "panel_d_content_audit",
        "panel_c_content_audit",
        "panel_c_message_level_content_audit",
        "module_4a_keyword_quality",
    ):
        if key in summary:
            return summary[key]
    raise KeyError("Summary JSON has no message-level content-audit results.")


def extract_latency(summary: Dict[str, Any]) -> Dict[str, Any]:
    if "panel_a_latency" in summary:
        return summary["panel_a_latency"]
    if "panel_d_latency" in summary:
        return summary["panel_d_latency"]

    if "module_3_latency" in summary:
        source = next(
            (item for item in summary["module_3_latency"] if item.get("arm_level_latency_stats")),
            None,
        )
        if source is not None:
            return source["arm_level_latency_stats"]

    raise KeyError("Summary JSON has no arm-level latency results.")


def load_operational_data(
    summary_json: Path,
    eps_drafts_edited: int,
    edit_categories: Dict[str, int],
) -> Dict[str, Any]:
    summary = load_json(summary_json)
    quality = extract_quality(summary)
    latency_stats = extract_latency(summary)

    feedback_counts = {
        "Human": int(quality["human_n_messages"]),
        "EPS-human": int(quality["eps_n_messages"]),
    }
    draft_total = feedback_counts["EPS-human"]
    if not 0 <= eps_drafts_edited <= draft_total:
        raise ValueError("--eps-drafts-edited must be between zero and the number of EPS drafts.")
    if sum(edit_categories.values()) != eps_drafts_edited:
        raise ValueError("The four edit category counts must sum to --eps-drafts-edited.")

    latency = {
        "Human": {
            "n": int(latency_stats["human"]["n"]),
            "median": float(latency_stats["human"]["median"]),
            "p25": float(latency_stats["human"]["p25"]),
            "p75": float(latency_stats["human"]["p75"]),
        },
        "EPS-human": {
            "n": int(latency_stats["eps"]["n"]),
            "median": float(latency_stats["eps"]["median"]),
            "p25": float(latency_stats["eps"]["p25"]),
            "p75": float(latency_stats["eps"]["p75"]),
        },
        "p_value": float(latency_stats["mann_whitney"]["p"]),
    }

    return {
        "feedback_counts": feedback_counts,
        "latency": latency,
        "draft_review": {
            "total": draft_total,
            "accepted_unchanged": draft_total - eps_drafts_edited,
            "edited": eps_drafts_edited,
            "edited_pct": eps_drafts_edited / draft_total * 100,
            "accepted_pct": (draft_total - eps_drafts_edited) / draft_total * 100,
            "categories": edit_categories,
        },
    }


def format_p(p_value: float) -> str:
    return r"$P$ < 0.001" if p_value < 0.001 else rf"$P$ = {p_value:.3f}"


def style_bar_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.tick_params(axis="both", labelsize=7.6, width=0.8, length=3, pad=2)
    ax.yaxis.grid(True, color="#D7DEE3", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)


def draw_feedback_count(ax: plt.Axes, data: Dict[str, Any]) -> None:
    counts = [data["feedback_counts"][arm] for arm in ARM_ORDER]
    x = np.arange(len(ARM_ORDER))
    ax.bar(
        x,
        counts,
        width=0.58,
        color=[ARM_COLORS[arm] for arm in ARM_ORDER],
        edgecolor=[ARM_EDGES[arm] for arm in ARM_ORDER],
        linewidth=0.75,
        alpha=0.88,
        zorder=3,
    )
    ax.set_title("Exercise feedback delivered", fontsize=9.0, pad=7)
    ax.set_ylabel("Instances, n", fontsize=8.0)
    ax.set_xticks(x, ARM_ORDER)
    ax.set_ylim(0, 500)
    style_bar_axis(ax)
    for xpos, value in zip(x, counts):
        ax.text(xpos, value + 500 * 0.035, f"{value}", ha="center", va="bottom", fontsize=8.2)

    increase = (counts[1] - counts[0]) / counts[0] * 100
    ax.text(0.5, 455, f"+{increase:.0f}%", ha="center", va="center", fontsize=8.2, fontweight="bold")


def draw_latency(ax: plt.Axes, data: Dict[str, Any]) -> None:
    latency = data["latency"]
    medians = [latency[arm]["median"] for arm in ARM_ORDER]
    p25 = [latency[arm]["p25"] for arm in ARM_ORDER]
    p75 = [latency[arm]["p75"] for arm in ARM_ORDER]
    yerr = np.asarray(
        [
            [median - low for median, low in zip(medians, p25)],
            [high - median for median, high in zip(medians, p75)],
        ]
    )
    x = np.arange(len(ARM_ORDER))
    ax.bar(
        x,
        medians,
        width=0.58,
        color=[ARM_COLORS[arm] for arm in ARM_ORDER],
        edgecolor=[ARM_EDGES[arm] for arm in ARM_ORDER],
        linewidth=0.75,
        alpha=0.88,
        zorder=3,
    )
    ax.errorbar(x, medians, yerr=yerr, fmt="none", ecolor="#1F1F1F", elinewidth=0.9, capsize=2.8, capthick=0.9, zorder=4)
    ax.set_title("Response latency", fontsize=9.0, pad=7)
    ax.set_ylabel("Median (IQR), min", fontsize=8.0)
    ax.set_xticks(
        x,
        [f"Human\nn={latency['Human']['n']}", f"EPS-human\nn={latency['EPS-human']['n']}"],
    )
    ax.set_ylim(0, 118)
    style_bar_axis(ax)
    for xpos, value, high in zip(x, medians, p75):
        label_y = value + 5 if value > 25 else high + 4
        ax.text(xpos - 0.11, label_y, f"{value:.1f}", ha="center", va="bottom", fontsize=8.2)

    ax.plot([0, 0, 1, 1], [101, 106, 106, 101], color="#252525", lw=0.75)
    ax.text(0.5, 109, format_p(latency["p_value"]), ha="center", va="bottom", fontsize=7.8)


def draw_draft_review(ax: plt.Axes, data: Dict[str, Any]) -> None:
    review = data["draft_review"]
    accepted_pct = review["accepted_pct"]
    edited_pct = review["edited_pct"]
    ax.barh([0], [accepted_pct], height=0.32, color=REVIEW_COLORS["Accepted unchanged"], edgecolor="#4C8FB3", linewidth=0.7, zorder=3)
    ax.barh([0], [edited_pct], left=[accepted_pct], height=0.32, color=REVIEW_COLORS["Edited before delivery"], edgecolor="#9A3B00", linewidth=0.7, zorder=3)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.72, 0.72)
    ax.set_yticks([])
    ax.set_xlabel("EPS drafts, %", fontsize=8.0)
    ax.set_title("Human review of EPS drafts", fontsize=9.0, pad=7)
    ax.tick_params(axis="x", labelsize=7.6, width=0.8, length=3, pad=2)
    ax.xaxis.grid(True, color="#D7DEE3", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.85)

    ax.text(accepted_pct / 2, 0, f"{review['accepted_unchanged']} unchanged\n({accepted_pct:.2f}%)", ha="center", va="center", fontsize=8.1)
    ax.text(99.0, 0.43, f"{review['edited']} edited\n({edited_pct:.2f}%)", ha="right", va="center", fontsize=8.1, color="#9A3B00", fontweight="bold")
    cats = review["categories"]
    category_text = (
        f"Edited drafts: {cats['Language polishing']} language, "
        f"{cats['Substantive content']} content, "
        f"{cats['Structural adjustment']} structure, "
        f"{cats['Safety-related']} safety"
    )
    ax.text(0, -0.48, category_text, ha="left", va="center", fontsize=6.9, color="#333333")


def write_summary_csv(path: Path, data: Dict[str, Any]) -> None:
    rows = []
    for arm in ARM_ORDER:
        rows.extend(
            [
                {"metric": "Exercise feedback delivered", "arm": arm, "value": data["feedback_counts"][arm], "unit": "instances"},
                {"metric": "Response latency median", "arm": arm, "value": f"{data['latency'][arm]['median']:.3f}", "unit": "minutes"},
                {"metric": "Response latency p25", "arm": arm, "value": f"{data['latency'][arm]['p25']:.3f}", "unit": "minutes"},
                {"metric": "Response latency p75", "arm": arm, "value": f"{data['latency'][arm]['p75']:.3f}", "unit": "minutes"},
            ]
        )
    review = data["draft_review"]
    rows.extend(
        [
            {"metric": "EPS draft total", "arm": "EPS-human", "value": review["total"], "unit": "drafts"},
            {"metric": "EPS drafts edited", "arm": "EPS-human", "value": review["edited"], "unit": "drafts"},
            {"metric": "EPS drafts edited", "arm": "EPS-human", "value": f"{review['edited_pct']:.3f}", "unit": "percent"},
        ]
    )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "arm", "value", "unit"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot operational outcomes in the weight-loss cohort (Fig. 2c).")
    parser.add_argument("--summary-json", type=Path, required=True, help="Summary JSON from enhanced_feedback_mediation.py.")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/clinical_trial"), help="Output directory.")
    parser.add_argument("--eps-drafts-edited", type=int, default=11, help="Number of EPS drafts edited before delivery (default: manuscript audit value, 11).")
    parser.add_argument("--language-edits", type=int, default=6)
    parser.add_argument("--content-edits", type=int, default=2)
    parser.add_argument("--structure-edits", type=int, default=2)
    parser.add_argument("--safety-edits", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    edit_categories = {
        "Language polishing": args.language_edits,
        "Substantive content": args.content_edits,
        "Structural adjustment": args.structure_edits,
        "Safety-related": args.safety_edits,
    }
    data = load_operational_data(args.summary_json, args.eps_drafts_edited, edit_categories)
    out_png = args.outdir / "operational_outcomes_panel_c.png"
    out_pdf = args.outdir / "operational_outcomes_panel_c.pdf"
    out_csv = args.outdir / "operational_outcomes_panel_c_summary.csv"
    write_summary_csv(out_csv, data)

    fig, axes = plt.subplots(1, 3, figsize=(9.3, 2.65), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.08, 1.38], "wspace": 0.42})
    draw_feedback_count(axes[0], data)
    draw_latency(axes[1], data)
    draw_draft_review(axes[2], data)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.28, top=0.82, wspace=0.43)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
