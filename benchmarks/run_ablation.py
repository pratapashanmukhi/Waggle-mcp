"""
benchmarks/run_ablation.py
===========================
Ablation study runner for Waggle's Recursive Context Assembly (RMCA).

Runs all seven ablation variants across benchmark families and scales,
computes per-component contribution deltas, and writes CSV / Markdown / JSON
output.

Usage
-----
  python benchmarks/run_ablation.py \\
    --variants rmca_full rmca_no_graph_expansion rmca_no_conflict_resolution \\
    --families pairwise codeqa context_reset \\
    --scales 128 512 \\
    --seed 42 \\
    --output benchmark_results/ \\
    --verbose

WARNING
-------
Results use deterministic synthetic data. Do not compare numerically to the
RLM paper until the exact public datasets and matching model setup are run.
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
from dataclasses import asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — works both from repo root and as installed package
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from waggle.recursive_context import AblationConfig, RecursiveContextController
from rlm_style_waggle_eval import (
    BenchResult,
    _make_graph,
    _BENCHMARK_RUNNERS,
    token_estimate,
    write_results,
    generate_pairwise_cases,
    run_pairwise_benchmark,
    generate_codeqa_cases,
    run_codeqa_benchmark,
    generate_context_reset_cases,
    run_context_reset_benchmark,
    _score_context_reset,
    exact_match,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ablation variant configurations
# ---------------------------------------------------------------------------

VARIANT_CONFIGS: dict[str, AblationConfig] = {
    "rmca_full":                   AblationConfig(),
    "rmca_no_decomposition":       AblationConfig(decompose=False),
    "rmca_no_graph_expansion":     AblationConfig(graph_expand=False),
    "rmca_no_conflict_resolution": AblationConfig(conflict_resolve=False),
    "rmca_no_verbatim_evidence":   AblationConfig(verbatim_evidence=False),
    "rmca_no_budget_compression":  AblationConfig(budget_compress=False),
    "rmca_random_subqueries":      AblationConfig(random_subqueries=True),
}

_ALL_VARIANTS = list(VARIANT_CONFIGS.keys())
_ALL_FAMILIES = ["pairwise", "codeqa", "context_reset"]

# ---------------------------------------------------------------------------
# Partial-run safety: flush results on unexpected exit
# ---------------------------------------------------------------------------

_partial_results: list[BenchResult] = []
_partial_output_dir: str = ""


def _flush_partial() -> None:
    if _partial_results and _partial_output_dir:
        out = Path(_partial_output_dir)
        out.mkdir(parents=True, exist_ok=True)
        partial_csv = out / "ablation_results_partial.csv"
        fieldnames = list(BenchResult.__dataclass_fields__.keys())
        with open(partial_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in _partial_results:
                writer.writerow(asdict(r))


atexit.register(_flush_partial)

# ---------------------------------------------------------------------------
# Core ablation runner
# ---------------------------------------------------------------------------


def _run_ablation_variant(
    graph: Any,
    query: str,
    token_budget: int,
    config: AblationConfig,
) -> tuple[str, float]:
    """Run a single ablation variant and return (context_pack, latency_ms)."""
    t0 = time.perf_counter()
    try:
        controller = RecursiveContextController(graph=graph)
        result = controller.build_context(
            query=query,
            token_budget=token_budget,
            depth=2,
            max_subqueries=6,
            mode="balanced",
            ablation=config,
        )
        pack = result.context_pack
    except Exception as exc:
        LOGGER.debug("ablation_variant failed: %s", exc)
        pack = ""
    latency = (time.perf_counter() - t0) * 1000
    return pack, latency


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------


def _compute_deltas(results: list[BenchResult]) -> list[BenchResult]:
    """Compute delta_vs_full and annotation for each result row."""
    # Build lookup: (family, scale) → full score
    full_scores: dict[tuple[str, int], float] = {}
    for r in results:
        if r.ablation_variant == "rmca_full":
            full_scores[(r.benchmark_family, r.scale_n)] = r.score

    component_map = {
        "rmca_no_decomposition":       "decompose",
        "rmca_no_graph_expansion":     "graph_expand",
        "rmca_no_conflict_resolution": "conflict_resolve",
        "rmca_no_verbatim_evidence":   "verbatim_evidence",
        "rmca_no_budget_compression":  "budget_compress",
        "rmca_random_subqueries":      "random_subqueries",
    }

    for r in results:
        key = (r.benchmark_family, r.scale_n)
        full_score = full_scores.get(key, 0.0)
        r.delta_vs_full = round(r.score - full_score, 4)
        component = component_map.get(r.ablation_variant, "")
        if component and r.delta_vs_full != 0.0:
            direction = "+" if r.delta_vs_full > 0 else ""
            r.annotation = (
                f"{component} responsible for "
                f"{direction}{r.delta_vs_full:.3f} on {r.benchmark_family}"
            )
    return results


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------


def run_ablation(
    variants: list[str],
    families: list[str],
    scales: list[int],
    token_budget: int,
    seed: int,
    output_dir: str,
    verbose: bool = False,
) -> list[BenchResult]:
    """
    Run ablation study for all (variant, family, scale) combinations.

    Returns a list of BenchResult rows with ablation_variant set.
    """
    global _partial_output_dir
    _partial_output_dir = output_dir

    all_results: list[BenchResult] = []

    for variant_name in variants:
        config = VARIANT_CONFIGS[variant_name]

        for family in families:
            for scale in scales:
                rng = random.Random(seed)

                db_path = f"/tmp/waggle_ablation_{variant_name}_{family}_{scale}.db"
                # Remove stale DB for a fresh run
                db_file = Path(db_path)
                if db_file.exists():
                    db_file.unlink()

                if verbose:
                    print(f"\n[ablation] variant={variant_name} family={family} scale={scale}")

                try:
                    graph = _make_graph(db_path)

                    # Generate cases using the appropriate generator
                    if family == "pairwise":
                        cases = generate_pairwise_cases(graph, scale_n=scale, rng=rng)
                        for case in cases:
                            pack, latency = _run_ablation_variant(
                                graph, case.question, token_budget, config
                            )
                            pack_lower = pack.lower()
                            # Scoring: check how many gold conflict pair labels appear in pack
                            found_conflict_labels = [
                                label for label, _ in case.gold_conflict_pairs
                                if label.lower() in pack_lower
                            ]
                            from rlm_style_waggle_eval import pairwise_f1
                            pred_pairs = [
                                (label, case.gold_conflict_pairs[0][1])
                                for label in found_conflict_labels
                            ]
                            score = pairwise_f1(pred_pairs, case.gold_conflict_pairs)

                            result = BenchResult(
                                benchmark_family="pairwise",
                                scale_n=scale,
                                method="ablation",
                                score=score,
                                tokens_returned=token_estimate(pack),
                                latency_ms=round(latency, 1),
                                context_pack_tokens=token_estimate(pack),
                                seed=seed,
                                token_budget=token_budget,
                                ablation_variant=variant_name,
                            )
                            all_results.append(result)
                            _partial_results.append(result)

                            if verbose:
                                print(f"  score={score:.3f} tokens={token_estimate(pack)}")

                    elif family == "codeqa":
                        cases = generate_codeqa_cases(graph, scale_n=scale, rng=rng)
                        for case in cases:
                            pack, latency = _run_ablation_variant(
                                graph, case.question, token_budget, config
                            )
                            score = exact_match(pack, "recursive_context.py")

                            result = BenchResult(
                                benchmark_family="codeqa",
                                scale_n=scale,
                                method="ablation",
                                score=score,
                                tokens_returned=token_estimate(pack),
                                latency_ms=round(latency, 1),
                                context_pack_tokens=token_estimate(pack),
                                seed=seed,
                                token_budget=token_budget,
                                ablation_variant=variant_name,
                            )
                            all_results.append(result)
                            _partial_results.append(result)

                            if verbose:
                                print(f"  score={score:.3f} tokens={token_estimate(pack)}")

                    elif family == "context_reset":
                        cases = generate_context_reset_cases(
                            graph, scale_n=scale, rng=rng, difficulty="easy"
                        )
                        for case in cases:
                            pack, latency = _run_ablation_variant(
                                graph, case.question, token_budget, config
                            )
                            scoring = _score_context_reset(pack, case, graph)
                            score = (
                                scoring["decision_recall"]
                                + scoring["constraint_recall"]
                                + scoring["next_step_accuracy"]
                                + scoring["active_decision_preference"]
                            ) / 4.0

                            result = BenchResult(
                                benchmark_family="context_reset",
                                scale_n=scale,
                                method="ablation",
                                score=score,
                                tokens_returned=token_estimate(pack),
                                latency_ms=round(latency, 1),
                                context_pack_tokens=token_estimate(pack),
                                seed=seed,
                                token_budget=token_budget,
                                ablation_variant=variant_name,
                                notes=json.dumps(scoring),
                            )
                            all_results.append(result)
                            _partial_results.append(result)

                            if verbose:
                                print(f"  score={score:.3f} tokens={token_estimate(pack)}")

                    else:
                        LOGGER.warning("Unknown family %r — skipping", family)

                except Exception as exc:
                    LOGGER.error(
                        "Error in ablation variant=%s family=%s scale=%d: %s",
                        variant_name, family, scale, exc,
                    )
                    if verbose:
                        import traceback
                        traceback.print_exc()

    # Compute deltas vs rmca_full
    all_results = _compute_deltas(all_results)
    return all_results


# ---------------------------------------------------------------------------
# Multi-seed support
# ---------------------------------------------------------------------------


def run_ablation_multi_seed(
    variants: list[str],
    families: list[str],
    scales: list[int],
    token_budget: int,
    seeds: list[int],
    output_dir: str,
    verbose: bool = False,
) -> list[BenchResult]:
    """
    Run ablation for multiple seeds and return per-seed rows plus aggregated
    summary rows (seed=-1 with mean_score and std_score).
    """
    import statistics

    per_seed_results: list[BenchResult] = []

    for seed in seeds:
        if verbose:
            print(f"\n=== Seed {seed} ===")
        seed_results = run_ablation(
            variants=variants,
            families=families,
            scales=scales,
            token_budget=token_budget,
            seed=seed,
            output_dir=output_dir,
            verbose=verbose,
        )
        per_seed_results.extend(seed_results)

    # Aggregate: group by (benchmark_family, scale_n, ablation_variant)
    groups: dict[tuple[str, int, str], list[float]] = {}
    group_proto: dict[tuple[str, int, str], BenchResult] = {}

    for r in per_seed_results:
        key = (r.benchmark_family, r.scale_n, r.ablation_variant)
        groups.setdefault(key, []).append(r.score)
        group_proto[key] = r  # keep last row as prototype for aggregated row

    aggregated: list[BenchResult] = []
    for key, scores in groups.items():
        proto = group_proto[key]
        mean_s = round(sum(scores) / len(scores), 4)
        std_s = round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0
        agg = BenchResult(
            benchmark_family=proto.benchmark_family,
            scale_n=proto.scale_n,
            method=proto.method,
            score=mean_s,
            tokens_returned=proto.tokens_returned,
            latency_ms=proto.latency_ms,
            context_pack_tokens=proto.context_pack_tokens,
            seed=-1,
            token_budget=proto.token_budget,
            ablation_variant=proto.ablation_variant,
            delta_vs_full=proto.delta_vs_full,
            annotation=proto.annotation,
            mean_score=mean_s,
            std_score=std_s,
        )
        aggregated.append(agg)

    return per_seed_results + aggregated


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_ablation_results(results: list[BenchResult], output_dir: str) -> dict[str, str]:
    """
    Write ablation results to CSV, Markdown, and JSON.

    Files written:
      {output_dir}/ablation_results.csv
      {output_dir}/ablation_results.md
      {output_dir}/ablation_results.json
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "ablation_results.csv"
    md_path = out / "ablation_results.md"
    json_path = out / "ablation_results.json"

    # --- CSV ---
    fieldnames = list(BenchResult.__dataclass_fields__.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # --- Markdown ---
    md_lines = [
        "# RMCA Ablation Study Results",
        "",
        "> **Warning:** Results use deterministic synthetic data. "
        "Do not compare numerically to the RLM paper.",
        "",
        "| Benchmark family | Scale | Variant | Score | Delta vs full | Annotation | Tokens |",
        "|---|---:|---|---:|---:|---|---:|",
    ]
    for r in results:
        delta_str = f"{r.delta_vs_full:+.4f}" if r.delta_vs_full != 0.0 else "0.0000"
        md_lines.append(
            f"| {r.benchmark_family} | {r.scale_n} | {r.ablation_variant} "
            f"| {r.score:.4f} | {delta_str} | {r.annotation} | {r.tokens_returned} |"
        )
    md_lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # --- JSON ---
    # Compute component_contributions: largest-impact component per family
    families = sorted({r.benchmark_family for r in results})
    component_contributions: dict[str, Any] = {}

    component_map = {
        "rmca_no_decomposition":       "decompose",
        "rmca_no_graph_expansion":     "graph_expand",
        "rmca_no_conflict_resolution": "conflict_resolve",
        "rmca_no_verbatim_evidence":   "verbatim_evidence",
        "rmca_no_budget_compression":  "budget_compress",
        "rmca_random_subqueries":      "random_subqueries",
    }

    for fam in families:
        fam_rows = [r for r in results if r.benchmark_family == fam and r.ablation_variant != "rmca_full"]
        if not fam_rows:
            continue
        # Find the variant with the largest absolute delta
        most_impactful = min(fam_rows, key=lambda r: r.delta_vs_full)
        component_contributions[fam] = {
            "most_impactful_variant": most_impactful.ablation_variant,
            "component": component_map.get(most_impactful.ablation_variant, ""),
            "delta_vs_full": most_impactful.delta_vs_full,
            "annotation": most_impactful.annotation,
        }

    summary: dict[str, Any] = {
        "warning": (
            "Ablation results use deterministic synthetic data. "
            "Do not compare numerically to the RLM paper."
        ),
        "total_rows": len(results),
        "variants": _ALL_VARIANTS,
        "component_contributions": component_contributions,
        "results": [asdict(r) for r in results],
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "csv": str(csv_path),
        "markdown": str(md_path),
        "json": str(json_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RMCA ablation study runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=_ALL_VARIANTS,
        choices=_ALL_VARIANTS,
        help="Ablation variants to run (default: all 7)",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=_ALL_FAMILIES,
        choices=_ALL_FAMILIES,
        help="Benchmark families to run (default: pairwise codeqa context_reset)",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=[128, 512],
        help="Memory sizes to test (default: 128 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42); overridden by --seeds",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Space-separated list of seeds for multi-seed runs; overrides --seed",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=1200,
        help="Token budget for context assembly (default: 1200)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results",
        help="Output directory for results files (default: benchmark_results/)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Verbose output",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    seeds = args.seeds if args.seeds else [args.seed]

    print("RMCA Ablation Study Runner")
    print(f"  variants     : {args.variants}")
    print(f"  families     : {args.families}")
    print(f"  scales       : {args.scales}")
    print(f"  seeds        : {seeds}")
    print(f"  token_budget : {args.token_budget}")
    print(f"  output       : {args.output}")
    print()
    print("WARNING: Results use synthetic data. Do not compare to RLM paper numerically.")
    print()

    if len(seeds) > 1:
        results = run_ablation_multi_seed(
            variants=args.variants,
            families=args.families,
            scales=args.scales,
            token_budget=args.token_budget,
            seeds=seeds,
            output_dir=args.output,
            verbose=args.verbose,
        )
    else:
        results = run_ablation(
            variants=args.variants,
            families=args.families,
            scales=args.scales,
            token_budget=args.token_budget,
            seed=seeds[0],
            output_dir=args.output,
            verbose=args.verbose,
        )

    if not results:
        print("No results produced.", file=sys.stderr)
        return 1

    paths = write_ablation_results(results, args.output)
    print("Ablation results written to:")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
