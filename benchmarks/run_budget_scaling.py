"""
benchmarks/run_budget_scaling.py
==================================
Token-budget scaling study for RMCA (Recursive Memory Context Assembly).

Sweeps over a range of token budgets and benchmark families to measure how
score, evidence coverage, latency, and tokens returned change as the budget
grows.

Usage:
  python benchmarks/run_budget_scaling.py \\
    --budgets 250 500 1000 2000 4000 \\
    --families context_reset pairwise linear_agg codeqa \\
    --methods raw_context query_graph build_context \\
    --scales 128 \\
    --seed 42 \\
    --output benchmark_results/
"""
from __future__ import annotations

import argparse
import atexit
import csv
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — same pattern as run_ablation.py
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rlm_style_waggle_eval import (
    BenchResult,
    _make_graph,
    _METHOD_RUNNERS,
    token_estimate,
    write_results,
    run_pairwise_benchmark,
    run_linear_agg_benchmark,
    run_codeqa_benchmark,
    run_context_reset_benchmark,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Partial-run safety (same pattern as run_ablation.py)
# ---------------------------------------------------------------------------

_partial_rows: list[dict] = []
_partial_csv_path: str = ""


def _flush_partial() -> None:
    if _partial_rows and _partial_csv_path:
        Path(_partial_csv_path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(_partial_rows[0].keys()) if _partial_rows else []
        with open(_partial_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(_partial_rows)


atexit.register(_flush_partial)

# ---------------------------------------------------------------------------
# Family → runner mapping
# ---------------------------------------------------------------------------

_FAMILY_RUNNERS = {
    "pairwise":      run_pairwise_benchmark,
    "linear_agg":    run_linear_agg_benchmark,
    "codeqa":        run_codeqa_benchmark,
    "context_reset": run_context_reset_benchmark,
}

_DEFAULT_BUDGETS  = [250, 500, 1000, 2000, 4000]
_DEFAULT_FAMILIES = ["context_reset", "pairwise", "linear_agg", "codeqa"]
_DEFAULT_METHODS  = ["raw_context", "query_graph", "build_context"]

# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_budget_scaling(
    budgets: list[int],
    families: list[str],
    methods: list[str],
    scales: list[int],
    seed: int,
    output_dir: str,
    verbose: bool = False,
) -> list[BenchResult]:
    """
    Sweep over (budget, family, scale, method) combinations.

    For each combination a fresh DB is created so results are independent.
    Partial results are flushed to CSV on exit via atexit.
    """
    global _partial_rows, _partial_csv_path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _partial_csv_path = str(out / "budget_scaling_partial.csv")

    all_results: list[BenchResult] = []

    for budget in budgets:
        for family in families:
            runner = _FAMILY_RUNNERS.get(family)
            if runner is None:
                LOGGER.warning("Unknown benchmark family: %s — skipping", family)
                continue

            for scale in scales:
                db_path = f"/tmp/waggle_budget_{family}_{scale}_{budget}.db"
                # Remove stale DB for a clean run
                if Path(db_path).exists():
                    Path(db_path).unlink()

                rng = random.Random(seed)

                if verbose:
                    print(f"\n[budget={budget}] [{family}] scale={scale} db={db_path}")

                try:
                    rows: list[BenchResult] = runner(
                        db_path=db_path,
                        scale_n=scale,
                        methods=methods,
                        token_budget=budget,
                        rng=rng,
                        verbose=verbose,
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "budget=%d family=%s scale=%d failed: %s",
                        budget, family, scale, exc,
                    )
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    continue

                # Stamp token_budget on every result row
                for r in rows:
                    r.token_budget = budget

                all_results.extend(rows)

                # Accumulate partial rows for atexit flush
                for r in rows:
                    _partial_rows.append(asdict(r))

                if verbose:
                    for r in rows:
                        print(
                            f"  {r.method}: score={r.score:.3f} "
                            f"ev_cov={r.evidence_coverage:.3f} "
                            f"tokens={r.tokens_returned} "
                            f"latency={r.latency_ms:.0f}ms"
                        )

    return all_results


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------


def write_budget_scaling_results(
    results: list[BenchResult],
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Write budget-scaling results to CSV, Markdown, and JSON.

    The CSV includes a ``token_budget`` column (already in BenchResult).
    The Markdown table columns: Family | Scale | Method | Budget | Score |
    Ev. Coverage | Tokens returned | Latency.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path  = out / "budget_scaling_results.csv"
    md_path   = out / "budget_scaling_results.md"
    json_path = out / "budget_scaling_results.json"

    # --- CSV ---
    fieldnames = list(BenchResult.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # --- Markdown ---
    md_lines = [
        "# RMCA Token-Budget Scaling Results",
        "",
        "> **Warning:** Results use deterministic synthetic data. "
        "Do not compare numerically to the RLM paper.",
        "",
        "| Family | Scale | Method | Budget | Score | Ev. Coverage | Tokens returned | Latency |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(results, key=lambda x: (x.benchmark_family, x.scale_n, x.method, x.token_budget)):
        md_lines.append(
            f"| {r.benchmark_family} | {r.scale_n} | {r.method} | {r.token_budget} "
            f"| {r.score:.3f} | {r.evidence_coverage:.3f} "
            f"| {r.tokens_returned} | {r.latency_ms:.0f} |"
        )
    md_lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # --- JSON ---
    families = sorted({r.benchmark_family for r in results})
    budgets  = sorted({r.token_budget for r in results})
    methods  = sorted({r.method for r in results})

    summary: dict[str, Any] = {
        "warning": (
            "Budget-scaling results use deterministic synthetic data. "
            "Do not compare numerically to the RLM paper."
        ),
        "total_rows": len(results),
        "budgets": budgets,
        "families": {},
    }
    for fam in families:
        summary["families"][fam] = {}
        for method in methods:
            summary["families"][fam][method] = {}
            for budget in budgets:
                matching = [
                    r for r in results
                    if r.benchmark_family == fam
                    and r.method == method
                    and r.token_budget == budget
                ]
                if matching:
                    r = matching[0]
                    summary["families"][fam][method][budget] = {
                        "score": r.score,
                        "evidence_coverage": r.evidence_coverage,
                        "tokens_returned": r.tokens_returned,
                        "latency_ms": r.latency_ms,
                    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "csv":      str(csv_path),
        "markdown": str(md_path),
        "json":     str(json_path),
    }


# ---------------------------------------------------------------------------
# Chart generation (4 charts)
# ---------------------------------------------------------------------------


def _generate_charts(results: list[BenchResult], output_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib not available — skipping chart generation. "
            "Install with: pip install matplotlib"
        )
        return []

    # Try to import style constants from plot_rlm_results
    try:
        sys.path.insert(0, str(_HERE))
        from plot_rlm_results import METHOD_COLORS, METHOD_MARKERS, METHOD_LABELS
    except ImportError:
        METHOD_COLORS = {}
        METHOD_MARKERS = {}
        METHOD_LABELS = {}

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    families = sorted({r.benchmark_family for r in results})
    methods  = sorted({r.method for r in results})
    budgets  = sorted({r.token_budget for r in results if r.token_budget > 0})

    metrics = [
        ("score",             "Score",            "score_vs_budget.png"),
        ("evidence_coverage", "Evidence Coverage", "evidence_coverage_vs_budget.png"),
        ("latency_ms",        "Latency (ms)",      "latency_vs_budget.png"),
        ("tokens_returned",   "Tokens Returned",   "tokens_returned_vs_budget.png"),
    ]

    for metric_field, metric_label, filename in metrics:
        fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 4))
        if len(families) == 1:
            axes = [axes]

        for ax, family in zip(axes, families):
            for method in methods:
                xs: list[int] = []
                ys: list[float] = []
                for budget in budgets:
                    matching = [
                        r for r in results
                        if r.benchmark_family == family
                        and r.method == method
                        and r.token_budget == budget
                    ]
                    if matching:
                        xs.append(budget)
                        ys.append(getattr(matching[0], metric_field, 0.0))
                if xs:
                    color  = METHOD_COLORS.get(method, "#888")
                    marker = METHOD_MARKERS.get(method, "o")
                    label  = METHOD_LABELS.get(method, method)
                    ax.plot(
                        xs, ys,
                        marker=marker,
                        color=color,
                        label=label,
                        linewidth=2,
                        markersize=6,
                    )

            ax.set_title(family, fontsize=10, fontweight="bold")
            ax.set_xlabel("Token budget", fontsize=9)
            ax.set_ylabel(metric_label, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(f"{metric_label} vs Token Budget", fontsize=12, fontweight="bold")
        out_path = charts_dir / filename
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        generated.append(out_path)

    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RMCA token-budget scaling study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=_DEFAULT_BUDGETS,
        help="Token budgets to sweep (default: 250 500 1000 2000 4000)",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=_DEFAULT_FAMILIES,
        choices=list(_FAMILY_RUNNERS.keys()),
        help="Benchmark families to evaluate",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=_DEFAULT_METHODS,
        choices=list(_METHOD_RUNNERS.keys()),
        help="Retrieval methods to compare",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=[128],
        help="Memory scales to test (default: 128)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic data generation",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Multiple seeds for statistical robustness (e.g. --seeds 42 43 44)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results",
        help="Output directory for results files",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    seeds = args.seeds if args.seeds else [args.seed]

    print("RMCA Token-Budget Scaling Study")
    print(f"  budgets  : {args.budgets}")
    print(f"  families : {args.families}")
    print(f"  methods  : {args.methods}")
    print(f"  scales   : {args.scales}")
    print(f"  seeds    : {seeds}")
    print(f"  output   : {args.output}")
    print()
    print("WARNING: Results use synthetic data. Do not compare to RLM paper numerically.")
    print()

    all_results: list[BenchResult] = []

    for seed in seeds:
        if len(seeds) > 1:
            print(f"Running seed {seed}...")

        seed_results = run_budget_scaling(
            budgets=args.budgets,
            families=args.families,
            methods=args.methods,
            scales=args.scales,
            seed=seed,
            output_dir=args.output,
            verbose=args.verbose,
        )
        all_results.extend(seed_results)

    if not all_results:
        print("No results produced.", file=sys.stderr)
        return 1

    paths = write_budget_scaling_results(all_results, args.output)
    print("Budget-scaling results written to:")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")

    # Generate charts (gracefully skipped if matplotlib unavailable)
    charts = _generate_charts(all_results, Path(args.output))
    if charts:
        print(f"\nGenerated {len(charts)} charts:")
        for p in charts:
            print(f"  {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
