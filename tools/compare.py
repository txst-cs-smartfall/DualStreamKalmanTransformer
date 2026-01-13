#!/usr/bin/env python
"""
Experiment comparison CLI with rich terminal output.

Usage:
    python tools/compare.py exp1 exp2 exp3
    python tools/compare.py --model kalman* --last 5
    python tools/compare.py --tags ablation encoder --format markdown
    python tools/compare.py exp1 exp2 --significance
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import fnmatch
from typing import List, Optional

from fusionlib.results import ExperimentManifest, ExperimentComparison


def print_table(df, title: str = "Experiment Comparison"):
    """Print comparison table with optional rich formatting."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=title, show_lines=False)

        # Add columns
        table.add_column("Experiment", style="cyan", no_wrap=True)
        table.add_column("Model", style="blue")
        table.add_column("Test F1", style="green", justify="right")
        table.add_column("Test Acc", justify="right")
        table.add_column("Val F1", justify="right")
        table.add_column("Gap", justify="right")
        table.add_column("Folds", justify="right")

        # Find best F1
        best_idx = df['test_f1_mean'].idxmax() if not df.empty else None

        for idx, row in df.iterrows():
            model_short = row['model'].split('.')[-1][:18] if '.' in str(row['model']) else str(row['model'])[:18]
            exp_short = row['experiment_id'][:22]

            # Highlight best
            is_best = idx == best_idx
            f1_str = f"[bold green]{row['test_f1_mean']:.2f} +/- {row['test_f1_std']:.2f}[/]" if is_best else f"{row['test_f1_mean']:.2f} +/- {row['test_f1_std']:.2f}"

            table.add_row(
                exp_short,
                model_short,
                f1_str if is_best else f"{row['test_f1_mean']:.2f} +/- {row['test_f1_std']:.2f}",
                f"{row['test_acc_mean']:.2f}",
                f"{row['val_f1_mean']:.2f}",
                f"{row['overfitting_gap']:.2f}",
                str(int(row['num_folds']))
            )

        console.print(table)

        if best_idx is not None:
            best = df.loc[best_idx]
            console.print(f"\n[bold green]Best:[/] {best['experiment_id']} (F1: {best['test_f1_mean']:.2f}%)")

    except ImportError:
        # Fallback to simple print
        print(f"\n{title}")
        print("=" * 80)
        print(f"{'Experiment':<25} {'Model':<18} {'Test F1':>15} {'Acc':>8} {'Gap':>6}")
        print("-" * 80)

        for _, row in df.iterrows():
            model_short = row['model'].split('.')[-1][:18] if '.' in str(row['model']) else str(row['model'])[:18]
            exp_short = row['experiment_id'][:25]
            print(f"{exp_short:<25} {model_short:<18} {row['test_f1_mean']:>6.2f} +/- {row['test_f1_std']:<6.2f} {row['test_acc_mean']:>8.2f} {row['overfitting_gap']:>6.2f}")

        if not df.empty:
            best_idx = df['test_f1_mean'].idxmax()
            best = df.loc[best_idx]
            print(f"\nBest: {best['experiment_id']} (F1: {best['test_f1_mean']:.2f}%)")


def print_significance(result: dict, exp_a: str, exp_b: str):
    """Print statistical significance results."""
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        if 'error' in result:
            console.print(f"[red]Error: {result['error']}[/]")
            return

        sig_str = "[green]Yes[/]" if result.get('significant_005', False) else "[red]No[/]"
        sig_001 = "[green]Yes[/]" if result.get('significant_001', False) else "[red]No[/]"

        text = f"""Comparing: {exp_a} vs {exp_b}
Subjects: {result.get('n_subjects', 0)}
Mean A: {result.get('exp_a_mean', 0):.2f}  Mean B: {result.get('exp_b_mean', 0):.2f}
Difference: {result.get('mean_diff', 0):.2f}
t-statistic: {result.get('t_statistic', 'N/A')}
p-value: {result.get('p_value', 'N/A')}
Cohen's d: {result.get('cohens_d', 'N/A')}
Significant (p<0.05): {sig_str}
Significant (p<0.01): {sig_001}"""

        console.print(Panel(text, title="Statistical Comparison"))

    except ImportError:
        print(f"\nStatistical Comparison: {exp_a} vs {exp_b}")
        print("-" * 40)
        for k, v in result.items():
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare FusionTransformer experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/compare.py exp1 exp2 exp3
    python tools/compare.py --model kalman* --last 5
    python tools/compare.py --tags ablation --format markdown
    python tools/compare.py exp1 exp2 --significance
        """
    )
    parser.add_argument('experiments', nargs='*', help='Experiment IDs to compare')
    parser.add_argument('--model', '-m', type=str, help='Filter by model pattern (supports glob)')
    parser.add_argument('--tags', '-t', nargs='+', help='Filter by tags')
    parser.add_argument('--last', '-n', type=int, help='Compare last N experiments')
    parser.add_argument('--format', '-f', choices=['table', 'markdown', 'csv', 'json'],
                        default='table', help='Output format')
    parser.add_argument('--output', '-o', type=str, help='Save to file')
    parser.add_argument('--per-fold', action='store_true', help='Show per-fold breakdown')
    parser.add_argument('--significance', '-s', action='store_true',
                        help='Run statistical significance test (requires 2 experiments)')
    parser.add_argument('--experiments-dir', default='experiments',
                        help='Experiments directory (default: experiments)')
    parser.add_argument('--hard-subjects', type=int, metavar='N',
                        help='Show N hardest subjects')

    args = parser.parse_args()

    manifest = ExperimentManifest(experiments_dir=args.experiments_dir)
    comparison = ExperimentComparison(experiments_dir=args.experiments_dir)

    # Determine experiments to compare
    exp_ids: List[str] = []

    if args.experiments:
        # Direct experiment IDs (support glob patterns)
        all_exps = manifest.list_experiments(limit=1000)
        for pattern in args.experiments:
            for exp in all_exps:
                if fnmatch.fnmatch(exp['id'], pattern):
                    exp_ids.append(exp['id'])
        # Also try exact matches
        for exp_id in args.experiments:
            if manifest.exists(exp_id) and exp_id not in exp_ids:
                exp_ids.append(exp_id)
    else:
        # Use filters
        experiments = manifest.list_experiments(
            model=args.model,
            tags=args.tags,
            limit=args.last or 10
        )
        exp_ids = [e['id'] for e in experiments]

    if not exp_ids:
        print("No experiments found matching criteria.")
        print(f"\nManifest has {len(manifest)} experiments.")
        print("Use --last N to show recent experiments.")
        return

    print(f"Comparing {len(exp_ids)} experiments...\n")

    # Statistical significance (requires exactly 2 experiments)
    if args.significance:
        if len(exp_ids) != 2:
            print("Significance test requires exactly 2 experiments.")
            return
        result = comparison.statistical_comparison(exp_ids[0], exp_ids[1])
        print_significance(result, exp_ids[0], exp_ids[1])
        return

    # Generate comparison
    df = comparison.compare(exp_ids)

    if df.empty:
        print("No experiment data found. Check if metrics.json exists.")
        return

    # Output based on format
    if args.format == 'table':
        print_table(df)

    elif args.format == 'markdown':
        md = comparison.generate_markdown_table(exp_ids)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(md)
            print(f"Saved to: {args.output}")
        else:
            print(md)

    elif args.format == 'csv':
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Saved to: {args.output}")
        else:
            print(df.to_csv(index=False))

    elif args.format == 'json':
        if args.output:
            df.to_json(args.output, orient='records', indent=2)
            print(f"Saved to: {args.output}")
        else:
            print(df.to_json(orient='records', indent=2))

    # Per-fold breakdown
    if args.per_fold:
        print("\n" + "=" * 60)
        print("Per-Fold Comparison (Test F1)")
        print("=" * 60)
        pivot = comparison.per_fold_comparison(exp_ids)
        if not pivot.empty:
            print(pivot.to_string())

    # Hard subjects
    if args.hard_subjects:
        hard = comparison.find_hard_subjects(exp_ids, n=args.hard_subjects)
        print(f"\n{args.hard_subjects} Hardest Subjects:")
        for subj, f1 in hard:
            print(f"  {subj}: {f1:.2f}%")


if __name__ == '__main__':
    main()
