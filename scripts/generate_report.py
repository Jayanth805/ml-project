"""
Script to generate reports from experiment results.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import ensure_dir, load_json
from src.reporting import ReportGenerator, generate_experiment_summary


def generate_report(
    results_dir: str = "results",
    output_dir: str = "reports",
    report_type: str = "experiment"
):
    """
    Generate reports from experiment results.

    Args:
        results_dir: Directory containing experiment results
        output_dir: Output directory for reports
        report_type: Type of report to generate
    """
    print("=" * 60)
    print(f"Generating {report_type} Report")
    print("=" * 60)

    results_path = Path(results_dir)
    output_path = Path(output_dir)
    ensure_dir(output_path)

    # Load all results
    results = {}

    # Load individual model results
    for result_file in results_path.glob("*_results.json"):
        model_name = result_file.stem.replace("_results", "")
        try:
            results[model_name] = load_json(result_file)
            print(f"Loaded: {result_file.name}")
        except Exception as e:
            print(f"Failed to load {result_file.name}: {e}")

    # Load baseline results
    if (results_path / "baseline_results.json").exists():
        try:
            baseline = load_json(results_path / "baseline_results.json")
            results['baseline'] = baseline
            print("Loaded: baseline_results.json")
        except Exception as e:
            print(f"Failed to load baseline_results.json: {e}")

    # Load image swap results
    if (results_path / "image_swap_results.json").exists():
        try:
            swap_results = load_json(results_path / "image_swap_results.json")
            results['image_swap'] = swap_results
            print("Loaded: image_swap_results.json")
        except Exception as e:
            print(f"Failed to load image_swap_results.json: {e}")

    if not results:
        print("No results found!")
        return

    print(f"\nTotal results loaded: {len(results)}")

    # Generate reports
    print("\nGenerating reports...")

    generator = ReportGenerator(str(output_path))

    # Generate text report
    text_path = generator.generate_text_report(
        results,
        report_type=report_type,
        output_filename=f"{report_type}_report.txt"
    )
    print(f"Generated: {text_path}")

    # Generate PDF
    try:
        pdf_path = generator.generate_pdf_report(
            results,
            report_type=report_type,
            output_filename=f"{report_type}_report.pdf"
        )
        print(f"Generated: {pdf_path}")
    except Exception as e:
        print(f"PDF generation failed: {e}")

    # Export JSON
    json_path = generator.export_json(
        results,
        output_filename=f"{report_type}_results.json"
    )
    print(f"Generated: {json_path}")

    # Export metrics CSV
    if 'consistency_aware' in results:
        metrics_path = generator.export_metrics_csv(
            results['consistency_aware'],
            output_filename=f"consistency_aware_metrics.csv"
        )
        print(f"Generated: {metrics_path}")

    print("\n" + "=" * 60)
    print("Report generation complete!")
    print(f"Reports saved to: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate reports from experiment results")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing experiment results")
    parser.add_argument("--output-dir", default="reports",
                        help="Output directory for reports")
    parser.add_argument("--report-type", default="experiment",
                        choices=['experiment', 'evaluation', 'prediction'],
                        help="Type of report to generate")

    args = parser.parse_args()

    generate_report(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        report_type=args.report_type
    )


if __name__ == "__main__":
    main()
