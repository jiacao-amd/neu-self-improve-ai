"""Combine local evaluator summaries with Search-R1 paper reference scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).parent
PAPER = HERE / "results" / "paper_reference.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare measured Bamboogle summaries against paper targets."
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=SUMMARY.json",
        help="Evaluator summary and display label; repeat for each run.",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()

    rows: list[dict[str, str | int | float]] = []
    with PAPER.open(encoding="utf-8", newline="") as handle:
        for paper_row in csv.DictReader(handle):
            rows.append(
                {
                    "label": f"Paper: {paper_row['method']} ({paper_row['algorithm']})",
                    "source": "paper reference",
                    "method": paper_row["method"],
                    "algorithm": paper_row["algorithm"],
                    "examples": 125,
                    "correct": int(paper_row["correct_of_125"]),
                    "exact_match": float(paper_row["accuracy"]),
                }
            )

    for item in args.run:
        if "=" not in item:
            parser.error(f"--run must be LABEL=SUMMARY.json, got: {item}")
        label, raw_path = item.split("=", 1)
        summary_path = Path(raw_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "label": label,
                "source": "measured run",
                "method": summary.get("mode", "unknown"),
                "algorithm": label,
                "examples": int(summary["examples"]),
                "correct": int(summary["correct"]),
                "exact_match": float(summary["exact_match"]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            f"Wrote {csv_path}; install matplotlib to also create the chart."
        ) from exc

    labels = [str(row["label"]) for row in rows]
    scores = [float(row["exact_match"]) for row in rows]
    colors = ["#9ca3af" if row["source"] == "paper reference" else "#2563eb" for row in rows]
    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 1.05), 5.5))
    bars = ax.bar(range(len(rows)), scores, color=colors)
    ax.set_ylabel("Bamboogle exact match")
    ax.set_ylim(0, max(0.1, max(scores, default=0) * 1.2))
    ax.set_xticks(range(len(rows)), labels, rotation=35, ha="right")
    ax.set_title("Search-R1 Bamboogle results (paper references vs measured runs)")
    ax.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{float(row['exact_match']):.3f}\n{row['correct']}/{row['examples']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    png_path = args.output_dir / "comparison.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
