"""
Report generation utilities for creating PDF and CSV reports.
"""

import os
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .utils import ensure_dir, save_json, timestamp_now


class ReportGenerator:
    """
    Generate reports in various formats (PDF, CSV, JSON).
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.

        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = ensure_dir(output_dir)

    def generate_prediction_report(
        self,
        result: Dict[str, Any],
        output_filename: Optional[str] = None
    ) -> str:
        """
        Generate a prediction report.

        Args:
            result: Prediction result dictionary
            output_filename: Optional output filename

        Returns:
            Path to saved report
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"prediction_report_{timestamp}.txt"

        output_path = self.output_dir / output_filename

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("TRUTHLENS AI - PREDICTION REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Generated: {result.get('timestamp', timestamp_now())}\n\n")

            # Text preview
            f.write("TEXT PREVIEW:\n")
            f.write("-" * 40 + "\n")
            f.write(result.get('text_preview', 'N/A')[:500] + "\n\n")

            # Image info
            f.write("IMAGE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Filename: {result.get('image_filename', 'No image provided')}\n\n")

            # Prediction
            f.write("PREDICTION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Result: {result.get('prediction', 'N/A')}\n")
            f.write(f"Probability REAL: {result.get('probability_real', 0):.2%}\n")
            f.write(f"Probability FAKE: {result.get('probability_fake', 0):.2%}\n")
            f.write(f"Confidence: {result.get('confidence', 0):.2%}\n\n")

            # Consistency analysis
            if 'consistency_interpretation' in result:
                f.write("IMAGE-TEXT CONSISTENCY:\n")
                f.write("-" * 40 + "\n")
                interp = result['consistency_interpretation']
                f.write(f"CLIP Similarity Score: {result.get('clip_similarity', 0):.4f}\n")
                f.write(f"Interpretation: {interp.get('interpretation', 'N/A')}\n")
                f.write(f"Level: {interp.get('level', 'N/A')}\n\n")

            # Model info
            f.write("MODEL INFORMATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Model Type: {result.get('model_type', 'N/A')}\n")
            f.write(f"Model Version: {result.get('model_version', 'N/A')}\n\n")

            # Disclaimer
            f.write("=" * 60 + "\n")
            f.write("IMPORTANT DISCLAIMER\n")
            f.write("=" * 60 + "\n")
            f.write(result.get('disclaimer', ''))

        return str(output_path)

    def generate_pdf_report(
        self,
        results: Dict[str, Any],
        report_type: str = "evaluation",
        output_filename: Optional[str] = None
    ) -> str:
        """
        Generate a PDF report.

        Args:
            results: Results dictionary
            report_type: Type of report ('evaluation', 'experiment', 'prediction')
            output_filename: Optional output filename

        Returns:
            Path to saved PDF
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError:
            print("reportlab not installed, generating text report instead")
            return self.generate_text_report(results, report_type, output_filename)

        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{report_type}_report_{timestamp}.pdf"

        output_path = self.output_dir / output_filename

        # Create PDF
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=30
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=12
        )

        story = []

        # Title
        title = "TruthLens AI - " + report_type.replace('_', ' ').title() + " Report"
        story.append(Paragraph(title, title_style))
        story.append(Paragraph(f"Generated: {timestamp_now()}", styles['Normal']))
        story.append(Spacer(1, 20))

        if report_type == 'evaluation':
            self._add_evaluation_content(story, results, styles, heading_style)
        elif report_type == 'experiment':
            self._add_experiment_content(story, results, styles, heading_style)
        elif report_type == 'prediction':
            self._add_prediction_content(story, results, styles, heading_style)

        # Build PDF
        doc.build(story)

        return str(output_path)

    def _add_evaluation_content(
        self,
        story: List,
        results: Dict[str, Any],
        styles,
        heading_style
    ):
        """Add evaluation report content to PDF."""
        story.append(Paragraph("Evaluation Summary", heading_style))
        story.append(Spacer(1, 10))

        # Metrics table
        metrics = results.get('metrics', {})
        data = [
            ['Metric', 'Value'],
            ['Accuracy', f"{metrics.get('accuracy', 0):.4f}"],
            ['Macro F1', f"{metrics.get('macro_f1', 0):.4f}"],
            ['Weighted F1', f"{metrics.get('weighted_f1', 0):.4f}"],
            ['Precision', f"{metrics.get('precision', 0):.4f}"],
            ['Recall', f"{metrics.get('recall', 0):.4f}"],
            ['ROC-AUC', f"{metrics.get('roc_auc', 0):.4f}"],
            ['Samples', str(metrics.get('n_samples', 0))],
            ['Positive Samples', str(metrics.get('n_positive', 0))],
            ['Negative Samples', str(metrics.get('n_negative', 0))]
        ]

        table = Table(data, colWidths=[2.5*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

        # Confusion Matrix
        story.append(Paragraph("Confusion Matrix", heading_style))
        cm_data = results.get('confusion_matrix', [])
        if cm_data:
            cm_table = Table([
                ['', 'Predicted REAL', 'Predicted FAKE'],
                ['Actual REAL', cm_data[0][0], cm_data[0][1]],
                ['Actual FAKE', cm_data[1][0], cm_data[1][1]]
            ], colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
            cm_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(cm_table)
        story.append(Spacer(1, 20))

        # Disclaimer
        story.append(Paragraph("Limitations", heading_style))
        story.append(Paragraph(
            "• Results are based on the Fakeddit dataset which uses distant supervision labels.<br/>"
            "• CLIP similarity measures semantic alignment, not factual accuracy.<br/>"
            "• Model performance may vary on different news sources and domains.",
            styles['Normal']
        ))

    def _add_experiment_content(
        self,
        story: List,
        results: Dict[str, Any],
        styles,
        heading_style
    ):
        """Add experiment report content to PDF."""
        story.append(Paragraph("Experiment Configuration", heading_style))

        config = results.get('config', {})
        story.append(Paragraph(f"Model Type: {config.get('model_type', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"Dataset: {config.get('dataset', 'N/A')}", styles['Normal']))
        story.append(Spacer(1, 10))

        story.append(Paragraph("Results Summary", heading_style))

        if 'original' in results and 'swapped' in results:
            orig = results['original'].get('metrics', {})
            swap = results['swapped'].get('metrics', {})

            data = [
                ['Metric', 'Original', 'Image-Swapped', 'Difference'],
                ['Accuracy', f"{orig.get('accuracy', 0):.4f}",
                 f"{swap.get('accuracy', 0):.4f}",
                 f"{results.get('differences', {}).get('accuracy', 0):.4f}"],
                ['Macro F1', f"{orig.get('macro_f1', 0):.4f}",
                 f"{swap.get('macro_f1', 0):.4f}",
                 f"{results.get('differences', {}).get('macro_f1', 0):.4f}"],
                ['Precision', f"{orig.get('precision', 0):.4f}",
                 f"{swap.get('precision', 0):.4f}",
                 f"{results.get('differences', {}).get('precision', 0):.4f}"]
            ]

            table = Table(data, colWidths=[1.5*inch, 1.2*inch, 1.5*inch, 1.2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)

    def _add_prediction_content(
        self,
        story: List,
        results: Dict[str, Any],
        styles,
        heading_style
    ):
        """Add prediction report content to PDF."""
        story.append(Paragraph(f"Prediction: {results.get('prediction', 'N/A')}", heading_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph(
            f"Probability of FAKE: {results.get('probability_fake', 0):.2%}",
            styles['Normal']
        ))
        story.append(Paragraph(
            f"Probability of REAL: {results.get('probability_real', 0):.2%}",
            styles['Normal']
        ))
        story.append(Paragraph(
            f"CLIP Similarity: {results.get('clip_similarity', 0):.4f}",
            styles['Normal']
        ))
        story.append(Spacer(1, 20))

        story.append(Paragraph("Disclaimer", heading_style))
        story.append(Paragraph(results.get('disclaimer', ''), styles['Normal']))

    def generate_text_report(
        self,
        results: Dict[str, Any],
        report_type: str = "evaluation",
        output_filename: Optional[str] = None
    ) -> str:
        """
        Generate a text report as fallback.

        Args:
            results: Results dictionary
            report_type: Type of report
            output_filename: Optional output filename

        Returns:
            Path to saved report
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{report_type}_report_{timestamp}.txt"

        output_path = self.output_dir / output_filename

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"TRUTHLENS AI - {report_type.upper()} REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {timestamp_now()}\n\n")

            f.write(str(results))

        return str(output_path)

    def export_metrics_csv(
        self,
        results: Dict[str, Any],
        output_filename: Optional[str] = None
    ) -> str:
        """
        Export metrics to CSV.

        Args:
            results: Results dictionary
            output_filename: Optional output filename

        Returns:
            Path to saved CSV
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"metrics_{timestamp}.csv"

        output_path = self.output_dir / output_filename

        metrics = results.get('metrics', {})

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in metrics.items():
                writer.writerow([key, value])

        return str(output_path)

    def export_predictions_csv(
        self,
        results: Dict[str, Any],
        output_filename: Optional[str] = None
    ) -> str:
        """
        Export predictions to CSV.

        Args:
            results: Results dictionary
            output_filename: Optional output filename

        Returns:
            Path to saved CSV
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"predictions_{timestamp}.csv"

        output_path = self.output_dir / output_filename

        predictions = results.get('predictions', [])
        labels = results.get('labels', [])
        probabilities = results.get('probabilities', [])
        sample_ids = results.get('sample_ids', [])

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'label', 'prediction', 'probability', 'correct'])
            for i in range(len(predictions)):
                pred = predictions[i] if i < len(predictions) else 'N/A'
                label = labels[i] if i < len(labels) else 'N/A'
                prob = probabilities[i] if i < len(probabilities) else 'N/A'
                sample_id = sample_ids[i] if i < len(sample_ids) else f'sample_{i}'
                correct = 'N/A'
                if pred != 'N/A' and label != 'N/A':
                    correct = 'Yes' if pred == label else 'No'
                writer.writerow([sample_id, label, pred, prob, correct])

        return str(output_path)

    def export_json(
        self,
        results: Dict[str, Any],
        output_filename: Optional[str] = None
    ) -> str:
        """
        Export results to JSON.

        Args:
            results: Results dictionary
            output_filename: Optional output filename

        Returns:
            Path to saved JSON
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"results_{timestamp}.json"

        output_path = self.output_dir / output_filename
        save_json(results, output_path)

        return str(output_path)


def generate_experiment_summary(
    results: Dict[str, Dict[str, Any]],
    output_dir: str = "reports"
) -> Dict[str, str]:
    """
    Generate all report formats for experiment results.

    Args:
        results: Dictionary of experiment results
        output_dir: Output directory

    Returns:
        Dictionary mapping report type to file path
    """
    generator = ReportGenerator(output_dir)
    output_files = {}

    # Generate text report
    output_files['text'] = generator.generate_text_report(results, "experiment")

    # Generate JSON export
    output_files['json'] = generator.export_json(results)

    # Try PDF
    try:
        output_files['pdf'] = generator.generate_pdf_report(results, "experiment")
    except Exception as e:
        print(f"PDF generation failed: {e}")

    return output_files
