"""Build the repository's architecture and study guide PDF."""

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Preformatted
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "project_architecture_study_guide.pdf"
NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#2364AA")
PALE = colors.HexColor("#EAF2F8")
INK = colors.HexColor("#243447")
MUTED = colors.HexColor("#607080")


class GuideDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename), pagesize=A4, title="TruthLens AI Project Architecture Study Guide",
                         author="Project code analysis")
        frame = Frame(17 * mm, 17 * mm, A4[0] - 34 * mm, A4[1] - 34 * mm,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(NAVY)
        canvas.rect(0, h - 8 * mm, w, 8 * mm, stroke=0, fill=1)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(17 * mm, 10 * mm, "TruthLens AI | Architecture Study Guide")
        canvas.drawRightString(w - 17 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                          fontSize=25, leading=31, textColor=NAVY, alignment=TA_CENTER, spaceAfter=12))
styles.add(ParagraphStyle(name="CoverSub", parent=styles["Normal"], fontSize=12, leading=18,
                          textColor=MUTED, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=17, leading=21,
                          textColor=NAVY, spaceBefore=8, spaceAfter=8, keepWithNext=True))
styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=12, leading=15,
                          textColor=BLUE, spaceBefore=7, spaceAfter=4, keepWithNext=True))
styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9.3, leading=13,
                          textColor=INK, spaceAfter=5))
styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=8, leading=10.5,
                          textColor=INK, spaceAfter=2))
styles.add(ParagraphStyle(name="Cellx", parent=styles["BodyText"], fontSize=7.5, leading=9.5,
                          textColor=INK))
styles.add(ParagraphStyle(name="CellHead", parent=styles["BodyText"], fontName="Helvetica-Bold",
                          fontSize=7.5, leading=9.5, textColor=colors.white))
styles.add(ParagraphStyle(name="Diagram", fontName="Courier", fontSize=7.3, leading=9.2,
                          leftIndent=4, textColor=NAVY, backColor=PALE, borderPadding=8))


def P(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def bullet(text):
    return P("&#8226; " + text)


def grid(rows, widths, repeat=1):
    converted = [[P(str(x), "CellHead" if r < repeat else "Cellx") for x in row]
                 for r, row in enumerate(rows)]
    t = Table(converted, colWidths=widths, repeatRows=repeat, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D5E0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FA")]),
    ]))
    return t


def section(title, description=None):
    out = [P(title, "H1x")]
    if description:
        out.append(P(description))
    return out


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    story = []
    story += [Spacer(1, 28 * mm), P("TRUTHLENS AI", "CoverTitle"),
              P("Complete System Architecture & Project File Guide", "CoverSub"),
              Spacer(1, 6 * mm), P("A study guide to the code, model variants, training pipeline, evaluation, and saved results", "CoverSub"),
              Spacer(1, 18 * mm)]

    summary = [
        ["At a glance", "What the repository contains"],
        ["Task", "Binary classification of Fakeddit Reddit posts using text, images, or both."],
        ["Encoders", "BERT base uncased (768-d), ViT base patch16/224 (768-d), CLIP ViT-B/32 (512-d embeddings for similarity)."],
        ["Trainable part", "By default pretrained encoders are frozen; the projection layers and classification heads learn from this subset."],
        ["Local data", "2,000 processed samples; 1,400 train, 300 validation, 300 test; balanced labels in each split."],
        ["Study question", "Does adding cross-modal interaction and CLIP similarity improve classification over basic text-image fusion?"],
    ]
    story += [grid(summary, [35 * mm, 130 * mm]), Spacer(1, 12 * mm),
              P("Read in this order: data flow → model architecture → variant comparison → training/evaluation → file reference.", "CoverSub"),
              PageBreak()]

    story += section("1. System architecture", "The project has an offline research pipeline and a checkpoint-based inference application.")
    story += [P("End-to-end research flow", "H2x"),
              Preformatted("""Fakeddit metadata + images
          |
          v
scripts/fetch_fakeddit_subset.py (optional subset download)
          |
          v
scripts/prepare_dataset.py -> data/processed/{train,val,test}.csv
          |
          +-----------------------------+
          |                             |
          v                             v
  scripts/train_model.py        scripts/evaluate_model.py
          |                             ^
          v                             |
  src/train.py + Trainer                |
  frozen BERT / ViT / CLIP              |
  trainable task head                   |
          |                             |
          v                             |
artifacts/checkpoints/<model>/<model>_best.pt
                                        |
                                        v
                      results/*_results.json + results/plots/*
                                        |
                                        v
                             app.py (Streamlit UI)
""", styles["Diagram"]), Spacer(1, 5),
              P("Inference path", "H2x"),
              Preformatted(r"""User text + uploaded image
          |
          v
src/predict.py: NewsAnalyzer + src/preprocessing.py
          |                                 |
          v                                 v
BERT tokenizer / ViT processor       CLIP processor
          |                                 |
          v                                 v
BERT CLS (768) + ViT CLS (768)       CLIP cosine similarity (scalar)
          \                                  /
           +------ selected classifier head-+
                           |
                           v
                  logit -> sigmoid probability
                           |
                           v
                 REAL/FAKE + confidence + caveat
""", styles["Diagram"]), PageBreak()]

    story += section("2. Neural architecture", "BERT and ViT produce independent representations. CLIP provides a separate semantic alignment score for the consistency-aware model.")
    story += [Preformatted("""TEXT                                    IMAGE
headline / title                        RGB image
    |                                        |
BERT tokenizer                           ViT processor
    |                                        |
BERT-base encoder (frozen)               ViT-base encoder (frozen)
    | CLS: 768 dims                          | CLS: 768 dims
    +-------------------+--------------------+
                        |
             trainable projections
             Linear -> LayerNorm -> ReLU -> Dropout
                        |
             T: 256 dims     I: 256 dims
                 |               |
                 +-- |T-I| ------+  (256)
                 +-- T * I -----+  (256)
                        |
CLIP text + image -> normalized embeddings -> cosine similarity (1)
                        |
      [T, I, abs diff, product, CLIP score] = 1025 dims
                        |
             Linear 1025->256 -> ReLU -> Dropout
             Linear 256->128 -> ReLU -> Dropout
             Linear 128->1 -> logit
                        |
              BCEWithLogitsLoss in training
              sigmoid for probability at inference
""", styles["Diagram"]), Spacer(1, 6),
              P("For the consistency-aware fusion vector: 4 × 256 + 1 = 1,025 features. CLIP similarity is cosine similarity between normalized CLIP text and image embeddings. It measures semantic alignment, not whether a claim is true.")]
    story += [P("Important implementation detail", "H2x"),
              P("The actual ablation in <font name='Courier'>src/fusion.py</font> uses projected text, projected image, and absolute difference (3 × 256 = 768 inputs). It removes both the CLIP scalar and the element-wise product. This differs from a common interpretation of ‘ablation without CLIP only’; see the model comparison below."),
              PageBreak()]

    story += section("3. Models included", "All variants share pretrained representations where applicable, but their task heads consume different inputs.")
    models = [
        ["Variant", "Inputs and head", "Purpose / difference"],
        ["text_only", "BERT CLS (768) → MLP → logit", "Text baseline; does not require an image."],
        ["image_only", "ViT CLS (768) → MLP → logit", "Visual baseline; ignores text features."],
        ["standard", "BERT + ViT → separate 256-d projections → concatenate (512) → MLP", "Basic multimodal fusion. No explicit difference, product, or CLIP score."],
        ["consistency_aware", "Text, image, absolute difference, element product, CLIP cosine (1,025) → MLP", "Proposed full interaction model; explicitly represents alignment and disagreement."],
        ["ablation", "Text, image, absolute difference (768) → MLP", "Removes CLIP and element-wise product. It is a reduced interaction model, not a one-factor CLIP-only ablation."],
    ]
    story += [grid(models, [31 * mm, 63 * mm, 71 * mm]), Spacer(1, 7),
              P("MLP means multilayer perceptron, a feed-forward neural network. The multimodal heads use linear layers with ReLU activations and dropout; the output is one logit for the fake class.")]
    story += [P("Saved test-set results", "H2x"),
              P("The following are read from the repository’s existing results JSON files. They are not freshly reproduced by this report. Each is based on 300 balanced test examples."),
              grid([
                  ["Model", "Accuracy", "Macro-F1", "Fake precision", "Fake recall", "ROC-AUC"],
                  ["consistency_aware", "0.800", "0.799", "0.771", "0.853", "0.865"],
                  ["standard", "0.797", "0.796", "0.820", "0.760", "0.843"],
                  ["text_only", "0.760", "0.760", "0.771", "0.740", "0.822"],
                  ["ablation", "0.767", "0.765", "0.733", "0.840", "0.848"],
                  ["image_only", "0.713", "0.711", "0.758", "0.627", "0.760"],
              ], [33 * mm, 22 * mm, 22 * mm, 29 * mm, 25 * mm, 24 * mm]), Spacer(1, 5),
              P("The best reported score is consistency-aware, but its macro-F1 advantage over standard fusion is only about 0.003. That small difference should not be treated as conclusive without repeated runs, confidence intervals, and checks for data leakage or subgroup effects.", "Smallx"),
              PageBreak()]

    story += section("4. Where and how models are trained", "The executable local training path is a Python command-line script, with notebooks for guided and Colab workflows.")
    story += [P("Local command", "H2x"),
              Preformatted("python scripts/train_model.py --model consistency_aware --epochs 1 --batch-size 4", styles["Code"]),
              P("Select one of: text_only, image_only, standard, consistency_aware, ablation. The script reads <font name='Courier'>data/processed/train.csv</font> and <font name='Courier'>val.csv</font>, and saves checkpoints under <font name='Courier'>artifacts/checkpoints/&lt;model&gt;/</font>. The test split is held for evaluation by <font name='Courier'>scripts/evaluate_model.py</font>.")]
    story += [P("Training sequence", "H2x")]
    steps = [
        ["Step", "What happens"],
        ["1. Configure", "Load configs/experiment.yaml; set seed; select CUDA if available, otherwise CPU."],
        ["2. Load data", "Read pre-made train and validation CSVs; resolve image IDs to JPG paths."],
        ["3. Preprocess", "Tokenize text for BERT and CLIP; process images separately for ViT and CLIP."],
        ["4. Forward", "Frozen encoders create features. Selected task head maps features to one logit per sample."],
        ["5. Optimize", "BCEWithLogitsLoss; AdamW on parameters with requires_grad=True; default LR 1e-3, weight decay 1e-4."],
        ["6. Validate", "Compute validation metrics each epoch. ReduceLROnPlateau is stepped using macro-F1."],
        ["7. Save", "Best checkpoint is selected by validation macro-F1. Checkpoint stores model/optimizer/scheduler state, configs and history."],
        ["8. Evaluate", "Load checkpoint and test.csv; save metrics/predictions JSON and confusion/probability/CLIP/ROC plots."],
    ]
    story += [grid(steps, [31 * mm, 134 * mm]), Spacer(1, 6),
              P("Defaults in code: batch size 16, 5 epochs, dropout 0.3, gradient clipping at 1.0, seed 42, frozen encoders, no mixed precision. The README example overrides epochs and batch size. The default command-line training path does not use cached features despite the config/docs describing a cache.")]
    story += [P("Metrics used during training", "H2x"),
              P("Training optimizes binary cross entropy on logits. At each epoch, accuracy, precision, recall, F1, macro/weighted scores, and ROC-AUC are calculated. Checkpoint selection and learning-rate scheduling use validation macro-F1; early stopping checks validation loss. Note: in <font name='Courier'>Trainer.train()</font>, early-stopping patience is based on whether validation loss improves, while the best checkpoint is chosen by macro-F1."),
              PageBreak()]

    story += section("5. Metrics: how to read them")
    metric_rows = [
        ["Metric", "Meaning in this binary task", "Use"],
        ["Accuracy", "Fraction of all examples classified correctly.", "Overall correctness; can hide class imbalance."],
        ["Precision (positive=FAKE)", "Of predicted FAKE items, fraction labeled FAKE.", "High precision means fewer false accusations."],
        ["Recall (positive=FAKE)", "Of actual FAKE items, fraction detected.", "High recall means fewer missed fake-labeled items."],
        ["F1", "Harmonic mean of positive-class precision and recall.", "Balances false positives and false negatives for FAKE."],
        ["Macro-F1", "Mean of class-wise F1 scores, giving each class equal weight.", "Main model-selection metric in this code."],
        ["Weighted-F1", "Class-wise F1 weighted by support (number of examples).", "Useful for imbalanced test data."],
        ["ROC-AUC", "Ranking ability across thresholds, from false-positive to true-positive rates.", "Threshold-independent separability; 0.5 is chance-like."],
        ["Average precision", "Area under the precision-recall curve (reported in result JSON).", "Useful for positive-class ranking, especially with imbalance."],
    ]
    story += [grid(metric_rows, [32 * mm, 79 * mm, 54 * mm]), Spacer(1, 6),
              P("A logit is converted to a probability with sigmoid. The default decision threshold is 0.5: probability ≥ 0.5 becomes FAKE, otherwise REAL. The evaluator reports the threshold and stores predictions, labels, probabilities, confusion matrix, sample IDs, and CLIP similarities.")]
    story += [P("Image swap experiment", "H2x"),
              P("<font name='Courier'>src/image_swap.py</font> can replace images while retaining text and labels, then compare prediction changes. This is a separate robustness experiment. The standard <font name='Courier'>evaluate_model.py</font> command does not automatically run it, so the saved model results shown in this guide are ordinary held-out evaluations, not image-swap results.")]
    story += [P("Data interpretation caveat", "H2x"),
              P("The local subset labels class 0 as Fakeddit True and class 1 as all other categories. This is a noisy dataset-derived label, not independent fact-checking. CLIP measures semantic compatibility, not truth. Scores describe this dataset and split, not general news detection performance."), PageBreak()]

    story += section("6. Complete project file architecture", "The tree below lists project code, documentation, notebooks, configuration, and representative data/artifact structure. The many individual JPG files are grouped.")
    tree = """multimodal-fake-news/
|-- README.md                         Project overview, setup, commands, limitations
|-- requirements.txt                 Local Python dependency ranges
|-- requirements-colab.txt           Colab-specific dependency set
|-- .gitignore                       Generated files, checkpoints, caches and IDE exclusions
|-- app.py                           Streamlit application: analysis, performance, experiments, docs
|-- configs/
|   `-- experiment.yaml              Dataset, encoders, training and evaluation settings
|-- src/
|   |-- __init__.py                  Public package exports and version
|   |-- dataset.py                   Samples, dataset loading, image reads, splits, feature cache
|   |-- preprocessing.py             BERT/ViT/CLIP tokenization and image processing
|   |-- encoders.py                  BERT, ViT, CLIP and encoder ensemble implementations
|   |-- fusion.py                    Projection, fusion heads, five model variants
|   |-- classifier.py                Full model composition and forward pass
|   |-- train.py                     Trainer, loss, optimizer, scheduler and checkpoints
|   |-- predict.py                   NewsAnalyzer, result object, batch and ensemble inference
|   |-- evaluate.py                  Test loop, metrics aggregation, plots, comparisons
|   |-- metrics.py                   Classification metrics and model rankings
|   |-- image_swap.py                Controlled image replacement experiment
|   |-- plotting.py                  Training, comparison, ROC, ablation and swap plots
|   |-- reporting.py                 Text/PDF/CSV/JSON report utilities
|   `-- utils.py                     Seeds, config, device, JSON, paths, formatting helpers
|-- scripts/
|   |-- fetch_fakeddit_subset.py     Download metadata subset and selected images
|   |-- prepare_dataset.py           Map labels, verify images, make train/val/test CSVs
|   |-- train_model.py               CLI entry point for training one model variant
|   |-- evaluate_model.py            CLI entry point for held-out test evaluation
|   |-- generate_report.py           Build reports from saved result JSON files
|   `-- generate_study_guide.py      Rebuild this architecture study PDF
|-- docs/
|   |-- architecture.md              Encoder/fusion architecture reference
|   |-- methodology.md                Research question, experiments and caveats
|   |-- datasets.md                   Fakeddit and NewsCLIPpings notes and data prep
|   |-- vscode_colab_setup.md         Local and Colab setup directions
|   `-- troubleshooting.md            Common setup, data, training and app issues
|-- notebooks/
|   |-- 01_environment_check.ipynb    Check libraries/runtime/GPU
|   |-- 02_dataset_analysis.ipynb     Inspect dataset and class distributions
|   |-- 03_baseline_models.ipynb      Baseline model workflow
|   |-- 04_consistency_aware_training.ipynb  Proposed model training
|   |-- 05_image_swap_experiment.ipynb       Swap robustness workflow
|   |-- 06_evaluation_and_plots.ipynb         Metrics and visualization workflow
|   |-- COLAB_00_quickstart.ipynb    Colab setup and orientation
|   |-- COLAB_01_download_dataset.ipynb       Colab data acquisition
|   `-- COLAB_02_full_training.ipynb          Colab training workflow
|-- tests/
|   |-- test_dataset.py              Dataset, split and cache unit tests
|   |-- test_fusion_dimensions.py    Fusion shapes and trainable parameter tests
|   |-- test_metrics.py              Metric calculations and comparisons
|   `-- test_inference.py            Analyzer/model input and output checks
|-- data/
|   |-- fakeddit/fakeddit-train.parquet       Source metadata mirror (large)
|   |-- fakeddit/fakeddit_train_subset.tsv    Balanced sampled metadata
|   |-- fakeddit/images/*.jpg                Downloaded post images
|   `-- processed/{train,val,test}.csv       Prepared splits (1,400/300/300)
|-- artifacts/
|   `-- checkpoints/<variant>/*.pt           Best, epoch and final model checkpoints
|-- results/
|   |-- *_results.json               Saved test metrics and per-example outputs
|   `-- plots/*.png                  Confusion, probability, CLIP, ROC plots
|-- reports/
|   `-- prediction_report_*.txt      Previously generated prediction report"""
    story += [Preformatted(tree, styles["Code"]), PageBreak()]

    story += section("7. File-by-file usefulness", "This reference explains the role of every source, script, documentation, test, configuration, and notebook file. Data and generated artifact contents are described by category because there are hundreds of image/checkpoint files.")
    files = [
        ["File", "Use / when to read it"],
        ["README.md", "Start here for project goal, quick start, structure and stated limitations."],
        ["requirements.txt / requirements-colab.txt", "Install project libraries locally / in Colab."],
        ["app.py", "Launch the user-facing Streamlit tool; pages show analysis, saved performance, experiments and documentation."],
        ["configs/experiment.yaml", "Central defaults for columns, labels, pretrained model IDs, training and output paths."],
        ["src/__init__.py", "Convenience API: imports and re-exports commonly used project classes/functions."],
        ["src/dataset.py", "Read rows/images, represent samples, split datasets and optionally cache features."],
        ["src/preprocessing.py", "Convert strings and PIL images into model-ready tensors for BERT, ViT and CLIP."],
        ["src/encoders.py", "Load pretrained BERT/ViT/CLIP, freeze weights, extract embeddings and compute cosine similarity."],
        ["src/fusion.py", "Define projection layers, MLP heads and five fusion/baseline alternatives."],
        ["src/classifier.py", "Combine encoder ensemble with selected head; provide logits/probability interface."],
        ["src/train.py", "Training loop, BCE loss, AdamW, scheduler, validation, early stopping and checkpoint save/load."],
        ["src/predict.py", "Single/batch/ensemble inference and structured PredictionResult."],
        ["src/evaluate.py", "Held-out predictions, metric summary, image swap comparison hooks, and evaluation plots."],
        ["src/metrics.py", "Accuracy, precision/recall/F1, ROC-AUC, confusion matrix, bootstrap helpers and rankings."],
        ["src/image_swap.py", "Build mismatched image pairs and compare model behavior under controlled swaps."],
        ["src/plotting.py", "Reusable visualizations for histories, comparisons, ROC, CLIP similarity, ablation and swaps."],
        ["src/reporting.py", "Generate prediction/experiment reports and CSV/JSON exports."],
        ["src/utils.py", "Shared seed, config, device, filesystem, JSON, timestamp and class-weight helpers."],
        ["scripts/fetch_fakeddit_subset.py", "Sample balanced examples from the metadata mirror and fetch matching images."],
        ["scripts/prepare_dataset.py", "Create processed CSV files and stratified 70/15/15 split."],
        ["scripts/train_model.py", "Executable training CLI; chooses model and reads split files."],
        ["scripts/evaluate_model.py", "Executable test CLI; loads checkpoint and writes results/plots."],
        ["scripts/generate_report.py", "Aggregate existing result JSON files into human-readable reports."],
        ["scripts/generate_study_guide.py", "Rebuild this PDF from the project root."],
        ["docs/*.md", "Architecture, methodology, dataset, setup and troubleshooting explanations."],
        ["notebooks/*.ipynb", "Interactive runtime check, dataset analysis, model baselines, training, swaps and evaluation tutorials."],
        ["tests/*.py", "Unit-level checks for metrics, dataset/split/cache, fusion dimensions and inference contracts."],
        ["data/fakeddit/*", "Source/sample metadata and image files used to build the local subset."],
        ["data/processed/*.csv", "Prepared train, validation, test data; dataset summary records split counts."],
        ["artifacts/checkpoints/*/*.pt", "Serialized learned model weights plus training state/config/history."],
        ["results/*_results.json and results/plots/*", "Saved predictions/metrics and plots from prior model evaluations."],
        ["reports/prediction_report_*.txt", "Example output from an earlier prediction report."],
        [".gitignore", "Excludes environments, caches, checkpoints and selected generated outputs from version control."],
    ]
    story += [grid(files, [58 * mm, 107 * mm]), PageBreak()]

    story += section("8. How to study and reproduce", "A practical sequence for learning the repository without trying to understand every helper at once.")
    for x in [
        "Read docs/methodology.md and docs/architecture.md to understand the research question and intended model.",
        "Inspect configs/experiment.yaml, then scripts/fetch_fakeddit_subset.py and scripts/prepare_dataset.py to understand labels and split files.",
        "Trace one forward pass: src/preprocessing.py → src/encoders.py → src/classifier.py → src/fusion.py.",
        "Read src/train.py to see which parameters update, how validation drives checkpoint selection, and what gets serialized.",
        "Compare model variants in src/fusion.py, then inspect result JSONs and plots. Keep the 300-example test-set size in mind.",
        "Read src/predict.py and app.py to follow checkpoint loading and user input through inference.",
        "For reproducibility, record the dataset snapshot, image availability, seed, model checkpoint, package versions, and exact command. Do not infer image-swap results from ordinary evaluation JSONs."
    ]:
        story.append(bullet(x))
    story += [Spacer(1, 5), P("Reproduction commands", "H2x"),
              Preformatted("""python scripts/fetch_fakeddit_subset.py --per-class 1000
python scripts/prepare_dataset.py --csv data/fakeddit/fakeddit_train_subset.tsv --images data/fakeddit/images
python scripts/train_model.py --model consistency_aware --epochs 1 --batch-size 4
python scripts/evaluate_model.py --checkpoint artifacts/checkpoints/consistency_aware/consistency_aware_best.pt
streamlit run app.py""", styles["Code"]),
              P("The repository contains five roughly 1.4 GB best checkpoints and a 2,000-example local dataset. Training/inference also requires the Hugging Face pretrained model files (downloaded or cached separately). Missing/corrupt images can interrupt multimodal batches; check image availability before a full run."),
              P("Report provenance", "H2x"),
              P("This guide was generated from the checked-in source/docs, local processed split counts, and existing results JSON files. It describes repository behavior as implemented; comments/docs sometimes describe broader intentions than the CLI currently executes."),
              Spacer(1, 10), P("End of guide", "CoverSub")]

    doc = GuideDoc(OUT)
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    build()
