"""
TruthLens AI - Streamlit Web Application
Cross-Modal Consistency-Aware Multimodal Fake News Detection
"""

import os
import sys
import io
import json
import base64
import html
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.predict import NewsAnalyzer, PredictionResult
from src.utils import (
    load_config, ensure_dir, save_json, timestamp_now,
    format_probability, clip_similarity_to_consistency_score
)
from src.reporting import ReportGenerator
from src.metrics import compute_metrics

# Page configuration
st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Warm cream and terracotta editorial theme.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&display=swap');
    :root { --ink:#352a23; --muted:#79685a; --line:#e5d7c6; --terracotta:#a94f36; --terra-dark:#7e3829; --cream:#f5efe5; --paper:#fffaf2; --sage:#67745b; }
    .stApp { background:radial-gradient(ellipse at 5% 0%,#fffaf2 0,#f5efe5 53%,#eee3d4 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
    section.main > div.block-container { max-width:1440px; padding-top:2rem; padding-bottom:3rem; }
    [data-testid="stHeader"] { background:rgba(245,239,229,.94); }
    [data-testid="stSidebar"] { display:none; }
    h1,h2,h3,h4 { font-family:'Fraunces',Georgia,serif!important; letter-spacing:-.025em; color:var(--ink)!important; }
    p,li { color:#57483b; }
    [data-testid="stMetric"] { background:rgba(255,250,242,.92); border:1px solid var(--line); border-radius:17px; padding:17px 19px; box-shadow:0 10px 26px rgba(89,60,37,.06); }
    [data-testid="stMetricLabel"] p { color:var(--muted)!important; font-size:.8rem!important; letter-spacing:.04em; text-transform:uppercase; }
    [data-testid="stMetricValue"] { color:var(--terra-dark)!important; font-family:'Fraunces',Georgia,serif; }
    .stButton>button,.stDownloadButton>button { border-radius:11px; min-height:2.65rem; font-weight:700; border:1px solid #d9c7b4; color:var(--ink); background:#fffaf2; transition:all .18s ease; }
    .st-key-view_text_report button { color:#000!important; }
    .stButton>button[kind="primary"] { background:var(--terracotta); color:#fffaf2; border:1px solid var(--terracotta); }
    .stButton>button:hover,.stDownloadButton>button:hover { transform:translateY(-1px); border-color:var(--terracotta); color:var(--terra-dark); }
    [data-testid="stRadio"] [role="radiogroup"] { display:flex; flex-direction:row; flex-wrap:wrap; gap:5px; padding:5px; border:1px solid #e4d5c4; border-radius:999px; background:#eee3d4; }
    [data-testid="stRadio"] label { padding:7px 14px; border-radius:999px; color:#604f41!important; font-size:.9rem; }
    [data-testid="stRadio"] label:has(input:checked) { background:var(--terracotta); color:#fffaf2!important; box-shadow:0 3px 9px rgba(126,56,41,.2); }
    [data-testid="stRadio"] label:has(input:checked) p { color:#fffaf2!important; }
    input,textarea,[data-baseweb="select"]>div { background:#fffaf2!important; color:var(--ink)!important; border-color:#decebc!important; border-radius:10px!important; }
    [data-testid="stExpander"] { background:rgba(255,250,242,.76); border:1px solid var(--line); border-radius:14px; }
    [data-testid="stFileUploader"] section { background:#fffaf2; border:1px dashed #c78c72; border-radius:14px; }
    .app-masthead { padding:8px 0 20px; border-bottom:1px solid rgba(169,79,54,.16); margin-bottom:14px; }
    .masthead-kicker { color:var(--terracotta); font-size:.7rem; font-weight:700; letter-spacing:.2em; text-transform:uppercase; }
    .masthead-title { color:var(--ink); font-family:'Fraunces',Georgia,serif; font-size:2.2rem; font-weight:600; letter-spacing:-.04em; line-height:1.1; margin:5px 0 3px; }
    .masthead-title span { color:var(--terracotta); }
    .masthead-subtitle { color:var(--muted); font-size:.92rem; margin:0; }
    .hero { position:relative; overflow:hidden; padding:36px 40px; margin:8px 0 23px; border:1px solid #d8b8a2; border-radius:24px; background:linear-gradient(112deg,#f1dfca 0%,#f8ecdc 58%,#e9d8c2 100%); box-shadow:0 18px 46px rgba(89,60,37,.09); }
    .hero:after { content:''; position:absolute; width:280px; height:280px; right:-62px; top:-150px; border-radius:50%; border:1px solid rgba(169,79,54,.25); box-shadow:0 0 0 24px rgba(169,79,54,.045),0 0 0 50px rgba(169,79,54,.03); }
    .eyebrow { color:var(--terracotta); font-size:.7rem; letter-spacing:.19em; text-transform:uppercase; font-weight:700; }
    .hero h1 { margin:10px 0 8px; font-size:2.8rem; line-height:1.1; }
    .hero p { max-width:720px; margin:0; color:#655244; font-size:1.02rem; }
    .hero-meta { display:flex; gap:8px; flex-wrap:wrap; margin-top:21px; }
    .pill { display:inline-block; border:1px solid #d9c2ac; background:rgba(255,250,242,.62); color:#684637; border-radius:999px; padding:6px 11px; font-size:.76rem; }
    .section-label { margin:10px 0 14px; color:#49382d; font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:1.24rem; }
    .prediction-real { background:#e2eadb; color:#3e6347; border:1px solid #c8d9bd; }
    .prediction-fake { background:#f4dfd8; color:#873d2d; border:1px solid #e5b9ab; }
    .disclaimer { background:#f3e7ce; color:#6d5428; border:1px solid #e4d2aa; }
    .main-header { color:var(--ink); font-family:'Fraunces',Georgia,serif; }
    .sub-header { color:var(--muted); }
    @media(max-width:700px) { .hero {padding:26px 22px;} .hero h1 {font-size:2.15rem;} [data-testid="stRadio"] label {padding:6px 9px;} }
</style>
""", unsafe_allow_html=True)

# Model variants supported by the classifier and training script.
MODEL_OPTIONS = {
    "text_only": "Text only (BERT)",
    "image_only": "Image only (ViT)",
    "standard": "Standard fusion (BERT + ViT)",
    "consistency_aware": "Consistency aware (BERT + ViT + CLIP)",
    "ablation": "Ablation (fusion without CLIP similarity)",
}

MODEL_BRIEFINGS = {
    "text_only": ("Language patterns in the headline or article", "BERT encodes the text, then a classifier estimates REAL vs. FAKE.", "Text is required; an image is optional and is not used by this model."),
    "image_only": ("Visual patterns in the uploaded image", "ViT encodes the image, then a classifier estimates REAL vs. FAKE.", "An image is required. Text is not used to make this model's prediction."),
    "standard": ("Combined language and image features", "BERT and ViT feed the classifier; a separate low-CLIP rule can override the final label.", "Both text and an image are required."),
    "consistency_aware": ("Language, image features, and image-text semantic alignment", "BERT and ViT features are fused with CLIP similarity for classification.", "Both text and an image are required. CLIP measures semantic alignment, not truth."),
    "ablation": ("Combined language and image features without CLIP in its neural head", "BERT and ViT feed the classifier; a separate low-CLIP rule can override the final label.", "Both text and an image are required. This variant helps compare the role of CLIP."),
}

DEMO_TEXT = "The city council announced that public buses will be free every Sunday starting next month."


def clear_analysis_model():
    """Require an explicit reload after changing the selected model."""
    st.session_state.analyzer = None
    st.session_state.checkpoint_loaded = False
    st.session_state.last_result = None


def find_model_checkpoint(model_type: str) -> Path:
    """Find the conventional best-checkpoint path for a model variant."""
    checkpoint_root = Path(__file__).parent / "artifacts" / "checkpoints"
    candidates = [
        checkpoint_root / model_type / f"{model_type}_best.pt",
        checkpoint_root / f"{model_type}_best.pt",
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'checkpoint_loaded' not in st.session_state:
        st.session_state.checkpoint_loaded = False
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'results_history' not in st.session_state:
        st.session_state.results_history = []


@st.cache_resource
def load_analyzer(checkpoint_path: str, model_type: str):
    """Load the analyzer with cached checkpoint."""
    try:
        analyzer = NewsAnalyzer(checkpoint_path=checkpoint_path, model_type=model_type)
        if not analyzer.is_ready():
            raise ValueError("Checkpoint could not be loaded into the selected model")
        if analyzer.model_type != model_type:
            raise ValueError(f"Checkpoint contains {analyzer.model_type}, not {model_type}")
        return analyzer
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def get_download_link(file_path: str, link_text: str, mime_type: str = "application/pdf"):
    """Generate a download link for a file."""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    return href


def prediction_to_dict(result: PredictionResult) -> Dict[str, Any]:
    """Convert PredictionResult to dictionary for display."""
    return {
        'prediction': result.prediction,
        'probability_fake': result.probability_fake,
        'probability_real': result.probability_real,
        'confidence': result.confidence,
        'clip_similarity': result.clip_similarity,
        'consistency_interpretation': result.consistency_interpretation,
        'model_type': result.model_type,
        'model_version': result.model_version,
        'timestamp': result.timestamp,
        'text_preview': result.text_preview,
        'image_filename': result.image_filename,
        'classifier_prediction': result.classifier_prediction,
        'prediction_reason': result.prediction_reason,
        'disclaimer': result.disclaimer
    }


def main():
    """Main application entry point."""
    init_session_state()

    pages = {
        "Analyze": "analysis",
        "Performance": "performance",
        "Dataset Guide": "docs",
    }
    st.markdown("""
    <div class="app-masthead">
      <div class="masthead-kicker">Multimodal research studio</div>
      <div class="masthead-title">TruthLens <span>AI</span></div>
      <p class="masthead-subtitle">Explore model predictions, performance, and the dataset behind the analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    selected_page = st.radio(
        "Main navigation",
        list(pages.keys()),
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation",
    )

    # Load configuration
    config_path = Path(__file__).parent / "configs" / "experiment.yaml"
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        config = {}

    # Model checkpoint path
    checkpoint_dir = Path(__file__).parent / "artifacts" / "checkpoints"
    default_checkpoint = checkpoint_dir / "consistency_aware" / "consistency_aware_best.pt"

    # Page routing
    if pages[selected_page] == "analysis":
        show_analysis_page(config, default_checkpoint)
    elif pages[selected_page] == "performance":
        show_performance_page(config)
    elif pages[selected_page] == "docs":
        show_docs_page()


def show_analysis_page(config: Dict, default_checkpoint: Path):
    """News analysis page."""
    st.markdown('''<section class="hero"><div class="eyebrow">Multimodal intelligence / 01</div>
    <h1>TruthLens <span style="color:#a94f36">AI</span></h1>
    <p>Read the claim. Inspect the image. Surface signals that help you assess how well they fit together.</p>
    <div class="hero-meta"><span class="pill">BERT · language</span><span class="pill">ViT · vision</span><span class="pill">CLIP · alignment</span></div></section>''', unsafe_allow_html=True)

    selected_model_type = st.selectbox(
        "Select model",
        options=list(MODEL_OPTIONS),
        index=list(MODEL_OPTIONS).index("consistency_aware"),
        format_func=lambda model_type: MODEL_OPTIONS[model_type],
        key="analysis_model_type",
        on_change=clear_analysis_model,
        help="Choose a trained model variant for this analysis.",
    )
    focus, method, required = MODEL_BRIEFINGS[selected_model_type]
    with st.container(border=True):
        brief_cols = st.columns([1.05, 1.6, 1.6])
        brief_cols[0].markdown(f"**Model focus**  \n{focus}")
        brief_cols[1].markdown(f"**How it works**  \n{method}")
        brief_cols[2].markdown(f"**Input needed**  \n{required}")
    checkpoint_default = find_model_checkpoint(selected_model_type)

    summary_path = Path(__file__).parent / "data" / "processed" / "dataset_summary.json"
    try:
        dataset_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        dataset_summary = {}
    status_cols = st.columns(3)
    status_cols[0].metric("Selected model checkpoint", "READY" if checkpoint_default.is_file() else "NOT TRAINED")
    status_cols[1].metric("Prepared examples", f"{dataset_summary.get('total_samples', 0):,}")
    status_cols[2].metric("Compute device", "GPU" if torch.cuda.is_available() else "CPU")
    st.markdown("\n")

    # Model loading section
    with st.expander("⚙️ Model Settings", expanded=True):
        col1, col2 = st.columns([2, 1])

        checkpoint_key = f"analysis_checkpoint_path_{selected_model_type}"
        if not st.session_state.get(checkpoint_key) and checkpoint_default.is_file():
            st.session_state[checkpoint_key] = str(checkpoint_default)

        checkpoint_path = col1.text_input(
            "Checkpoint Path",
            value=str(checkpoint_default) if checkpoint_default.exists() else "",
            key=checkpoint_key,
            help=f"Checkpoint for {MODEL_OPTIONS[selected_model_type]}"
        )

        if col2.button("Load Model"):
            if checkpoint_path and os.path.exists(checkpoint_path):
                with st.spinner("Loading model..."):
                    st.session_state.analyzer = load_analyzer(checkpoint_path, selected_model_type)
                    if st.session_state.analyzer:
                        st.session_state.checkpoint_loaded = st.session_state.analyzer.is_ready()
                        if st.session_state.checkpoint_loaded:
                            st.success(f"Loaded {MODEL_OPTIONS[st.session_state.analyzer.model_type]}.")
                        else:
                            st.warning("Checkpoint structure may be incompatible.")
                    else:
                        st.session_state.checkpoint_loaded = False
            else:
                st.error(f"No checkpoint found for {MODEL_OPTIONS[selected_model_type]}. Train this model first or provide a valid checkpoint path.")

        if st.session_state.checkpoint_loaded:
            st.info(f"✅ Model loaded: {st.session_state.analyzer.model_type}")
        else:
            st.warning(f"No trained {MODEL_OPTIONS[selected_model_type]} model is loaded. Load its checkpoint before analyzing.")
            if not checkpoint_default.is_file():
                st.code(f"python scripts/train_model.py --model {selected_model_type}", language="bash")

    st.markdown("---")

    # Input section
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📝 Input Text")
        if st.button("Insert demo claim", key="insert_demo_claim", help="Adds a synthetic example for demonstrating the workflow."):
            st.session_state.analysis_text_input = DEMO_TEXT
        st.caption("Synthetic example for demonstrating the interface; it is not a verified news claim.")
        text_input = st.text_area(
            "News Headline or Article",
            height=200,
            placeholder="Enter a news headline or article text here...",
            help="Enter the text content of the news you want to analyze",
            key="analysis_text_input",
        )

    with col2:
        image_required = selected_model_type != "text_only"
        st.markdown(f"### 🖼️ Image ({'Required' if image_required else 'Optional'})")
        st.caption("Required for this model." if image_required else "Optional for the text-only model.")
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image associated with the news article"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    # Analysis button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

    analyze_clicked = col_btn1.button("🔍 Analyze News", type="primary", use_container_width=True)
    reset_clicked = col_btn2.button("🔄 Reset", use_container_width=True)

    # Handle reset
    if reset_clicked:
        st.session_state.last_result = None
        st.rerun()

    # Process analysis
    if analyze_clicked:
        if not st.session_state.checkpoint_loaded or not st.session_state.analyzer:
            st.error("Load a trained checkpoint for the selected model before analyzing.")
            return
        if selected_model_type != "image_only" and not text_input.strip():
            st.error("Please enter text for this model.")
            return
        if image_required and uploaded_file is None:
            st.error(f"The {MODEL_OPTIONS[selected_model_type]} model requires an image.")
            return

        with st.spinner("Analyzing..."):
            try:
                image_data = None
                if uploaded_file is not None:
                    image_data = Image.open(uploaded_file)

                result = st.session_state.analyzer.analyze(text_input, image_data)

                st.session_state.last_result = prediction_to_dict(result)
                st.session_state.results_history.append(prediction_to_dict(result))

            except Exception as e:
                st.error(f"Error during analysis: {e}")
                return

    # Display results
    if st.session_state.last_result:
        result = st.session_state.last_result

        st.markdown("---")
        st.markdown("### 📋 Analysis Results")

        if (selected_model_type in {"standard", "consistency_aware", "ablation"}
                and result["consistency_interpretation"].get("possible_mismatch")):
            st.error(
                "IMAGE-TEXT MISMATCH RULE APPLIED — final system label is FAKE. "
                f"The learned classifier alone predicted {result.get('classifier_prediction', 'N/A')}."
            )

        # Prediction display
        pred_class = result['prediction']
        if pred_class == "FAKE":
            st.markdown(f'<div class="prediction-fake">MODEL LABEL PREDICTION: FAKE</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-real">MODEL LABEL PREDICTION: REAL</div>', unsafe_allow_html=True)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Classifier probability REAL",
                f"{result['probability_real']:.1%}",
                help="Raw learned classifier probability before the image-text mismatch override"
            )
        with col2:
            st.metric(
                "Classifier probability FAKE",
                f"{result['probability_fake']:.1%}",
                help="Raw learned classifier probability before the image-text mismatch override"
            )
        with col3:
            st.metric(
                "Classifier confidence",
                f"{result['confidence']:.1%}",
                help="Confidence of the learned classifier before any mismatch override"
            )
        with col4:
            st.metric(
                "CLIP Similarity",
                f"{result['clip_similarity']:.3f}" if selected_model_type not in {"text_only", "image_only"} else "N/A",
                help=("CLIP is outside this neural head, but a separate low-alignment policy can override the final label."
                      if selected_model_type in {"standard", "ablation"} else
                      "CLIP similarity is an alignment diagnostic and classifier input; it does not verify truth."
                      if selected_model_type == "consistency_aware" else
                      "CLIP similarity is not applicable to this model.")
            )

        # Consistency interpretation
        st.markdown("#### 🔗 Image-Text Consistency")
        interp = result['consistency_interpretation']
        if selected_model_type in {"text_only", "image_only"}:
            interp = {
                "interpretation": "Not applicable to this single-modality model.",
                "level": "Not evaluated",
                "warning": None,
                "possible_mismatch": False,
            }

        consistency_col1, consistency_col2 = st.columns(2)
        with consistency_col1:
            st.markdown(f"**Interpretation:** {interp.get('interpretation', 'N/A')}")
            st.markdown(f"**Level:** {interp.get('level', 'N/A').replace('-', ' ').title()}")

        with consistency_col2:
            if interp.get('warning'):
                st.markdown(f"⚠️ {interp['warning']}")

        if selected_model_type in {"standard", "ablation"} and not interp.get("possible_mismatch"):
            st.info("CLIP is not part of this model's neural head, but the separate mismatch policy can override its label when similarity is below 0.22.")

        # Report generation
        st.markdown("---")
        st.markdown("### 📄 Export Report")

        export_col1, export_col2, export_col3, export_col4 = st.columns(4)

        # Prepare report
        report_data = result.copy()
        report_data['text'] = text_input
        if uploaded_file:
            report_data['image_name'] = uploaded_file.name

        # Text report
        with export_col1:
            if st.button("📝 View Text Report", use_container_width=True, key="view_text_report"):
                with st.expander("Report Content", expanded=True, key="report_content"):
                    report_text = "\n".join([
                        "TruthLens AI - Prediction Report",
                        f"Generated: {result['timestamp']}",
                        "-" * 40,
                        f"Prediction: {result['prediction']}",
                        f"Probability REAL: {result['probability_real']:.2%}",
                        f"Probability FAKE: {result['probability_fake']:.2%}",
                        f"CLIP Similarity: {result['clip_similarity']:.4f}",
                        "-" * 40,
                        "Disclaimer:",
                        result['disclaimer'],
                    ])
                    st.markdown(
                        f'<div style="color:#000 !important; white-space:pre-wrap;">{html.escape(report_text)}</div>',
                        unsafe_allow_html=True,
                    )

        # JSON export
        with export_col2:
            json_str = json.dumps(result, indent=2)
            st.download_button(
                "📥 Download JSON",
                data=json_str,
                file_name=f"prediction_{result['timestamp'][:19].replace(':', '')}.json",
                mime="application/json",
                use_container_width=True
            )

        # Print button
        with export_col3:
            if st.button("🖨️ Print Report", use_container_width=True):
                st.info("Use your browser's print function (Ctrl+P / Cmd+P) to print this page.")

        # PDF button
        with export_col4:
            if st.button("📄 Generate PDF", use_container_width=True):
                try:
                    reports_dir = Path(__file__).parent / "reports"
                    ensure_dir(reports_dir)
                    generator = ReportGenerator(str(reports_dir))
                    pdf_path = generator.generate_prediction_report(result)
                    st.success(f"PDF saved to: {pdf_path}")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

        # Disclaimer
        st.markdown("---")
        st.markdown(f'<div class="disclaimer">{result["disclaimer"]}</div>', unsafe_allow_html=True)

    elif not st.session_state.checkpoint_loaded:
        st.info("👆 Load a trained model checkpoint above to get predictions, or enter text above to see CLIP consistency analysis.")


def show_performance_page(config: Dict):
    """Model performance dashboard."""
    st.markdown('<p class="main-header">Model Performance Dashboard</p>', unsafe_allow_html=True)

    # Load results if available
    results_dir = Path(__file__).parent / "results"
    checkpoints_dir = Path(__file__).parent / "artifacts" / "checkpoints"

    # Check for saved results
    results_files = list(results_dir.glob("*_results.json"))

    if not results_files:
        st.info("No evaluation results found. Train models and run evaluations to see results here.")
        st.markdown("""
        ### Quick Start

        1. Open a Jupyter notebook from the `notebooks/` directory
        2. Run `04_consistency_aware_training.ipynb` to train models
        3. Run `06_evaluation_and_plots.ipynb` to evaluate and save results
        4. Return here to view results
        """)

        # Show example structure
        st.markdown("### Expected Results Structure")

        example_results = {
            "consistency_aware": {
                "accuracy": 0.85,
                "macro_f1": 0.84,
                "precision": 0.83,
                "recall": 0.85,
                "roc_auc": 0.91
            },
            "standard": {
                "accuracy": 0.82,
                "macro_f1": 0.81,
                "precision": 0.80,
                "recall": 0.82,
                "roc_auc": 0.88
            }
        }

        st.json(example_results)
        return

    # Load and display results
    st.markdown("### 📊 Model Comparison")

    all_results = {}
    for f in results_files:
        try:
            data = json.load(open(f))
            model_name = f.stem.replace("_results", "")
            all_results[model_name] = data.get('metrics', {})
        except:
            continue

    if all_results:
        # Display comparison table
        import pandas as pd

        metrics_to_show = ['accuracy', 'macro_f1', 'precision', 'recall', 'roc_auc']
        df_data = []

        for model_name, metrics in all_results.items():
            row = {'Model': model_name}
            for metric in metrics_to_show:
                row[metric.replace('_', ' ').title()] = metrics.get(metric, 0)
            df_data.append(row)

        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)

        # Visual comparison
        st.markdown("### 📈 Visual Comparison")

        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(all_results.keys())
        metrics = ['accuracy', 'macro_f1', 'precision', 'recall']
        x = np.arange(len(metrics))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            values = [all_results[model].get(m, 0) for m in metrics]
            offset = (i - len(models)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model)

        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.title() for m in metrics])
        ax.legend()
        ax.set_ylim([0, 1.1])

        st.pyplot(fig)

    # Check for plots
    plots_dir = results_dir / "plots"
    if plots_dir.exists():
        st.markdown("### 📉 Training Curves & Plots")
        plot_files = list(plots_dir.glob("*.png"))

        if plot_files:
            cols = st.columns(2)
            for i, plot_file in enumerate(plot_files[:6]):
                with cols[i % 2]:
                    st.image(str(plot_file), caption=plot_file.stem, use_container_width=True)
        else:
            st.info("No plots generated yet.")


def show_docs_page():
    """Present a visual guide to the local Fakeddit files and their formats."""
    project_root = Path(__file__).resolve().parent
    fakeddit_dir = project_root / "data" / "fakeddit"
    processed_dir = project_root / "data" / "processed"
    summary_path = processed_dir / "dataset_summary.json"

    st.markdown("""
    <div class="hero" style="margin-bottom:1.25rem">
      <div class="eyebrow">DATA / LOCAL FILES</div>
      <h1>Dataset <span style="color:#a94f36">Guide</span></h1>
      <p>How the Fakeddit text, images, labels, and prepared splits are organized in this project.</p>
    </div>
    """, unsafe_allow_html=True)

    summary = {}
    if summary_path.is_file():
        try:
            with summary_path.open("r", encoding="utf-8") as summary_file:
                summary = json.load(summary_file)
        except (OSError, json.JSONDecodeError):
            st.warning("The local dataset summary could not be read.")

    train_n = summary.get("train_samples", 1400)
    val_n = summary.get("val_samples", 300)
    test_n = summary.get("test_samples", 300)
    total_n = summary.get("total_samples", train_n + val_n + test_n)
    images_n = summary.get("images_available")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Examples", f"{total_n:,}")
    metric_cols[1].metric("Train / validation / test", f"{train_n:,} / {val_n:,} / {test_n:,}")
    metric_cols[2].metric("Images found", f"{images_n:,}" if images_n is not None else "See image folder")
    metric_cols[3].metric("Prepared labels", "0 = REAL | 1 = FAKE")

    st.markdown("### Storage locations")
    st.caption("These paths are resolved from the project folder on this computer.")
    location_col1, location_col2 = st.columns(2)
    with location_col1:
        st.markdown("**Original dataset and images**")
        st.code(str(fakeddit_dir), language=None)
    with location_col2:
        st.markdown("**Prepared training data**")
        st.code(str(processed_dir), language=None)

    st.markdown("### Files, one by one")
    st.caption("Open a file card to see its exact path, format, and purpose.")
    files = [
        ("1. Source metadata", fakeddit_dir / "fakeddit-train.parquet", "Apache Parquet", "Downloaded source table with Fakeddit post metadata. It is the source used to select this local sample."),
        ("2. Selected dataset subset", fakeddit_dir / "fakeddit_train_subset.tsv", "TSV (tab-separated text)", "The sampled examples, with clean_title, image_id, 3_way_label, and image_url fields."),
        ("3. Image collection", fakeddit_dir / "images", "Folder of JPEG images", "Images are named by ID. A row with image_id abc123 refers to images/abc123.jpg."),
        ("4. Combined processed table", processed_dir / "fakeddit_processed.csv", "CSV (comma-separated text)", "All prepared examples before the train, validation, and test split."),
        ("5. Training split", processed_dir / "train.csv", "CSV (comma-separated text)", f"Examples used to fit the model ({train_n:,} rows)."),
        ("6. Validation split", processed_dir / "val.csv", "CSV (comma-separated text)", f"Examples used during training to guide model selection ({val_n:,} rows)."),
        ("7. Test split", processed_dir / "test.csv", "CSV (comma-separated text)", f"Held-out examples used to estimate performance after training ({test_n:,} rows)."),
        ("8. Dataset summary", summary_path, "JSON", "Counts for the dataset and its splits, class balance, and image availability."),
    ]
    for title, file_path, file_format, description in files:
        with st.expander(title, expanded=False):
            st.markdown(f"**Format:** {file_format}")
            st.markdown(f"**Location:** `{file_path}`")
            st.markdown(description)
            if file_path.is_dir():
                st.caption("Individual image path pattern: " + str(file_path / "<image_id>.jpg"))

    st.info(
        "Yes, this project includes CSV files: fakeddit_processed.csv, train.csv, val.csv, and test.csv. "
        "It also includes one TSV subset file. There are no Excel workbooks (.xlsx or .xls) in the dataset folder. "
        "CSV and TSV are plain-text tables that can also be opened in Excel, but they are not Excel workbook files."
    )

    st.markdown("### How a prepared row is structured")
    st.code('text,image_id,label,image_exists\n"post title text",abc123,0,True', language="csv")
    st.markdown("""
    | Field | Example / type | Why it matters |
    |---|---|---|
    | `text` | Text string | The language model reads this title or post text. |
    | `image_id` | `abc123` | Connects the row to its JPEG image for visual and cross-modal analysis. |
    | `label` | `0` or `1` | The supervised target the classifier learns and evaluation measures. Here, 0 is the subset's true category and 1 groups the other categories. |
    | `image_exists` | `True` or `False` | Shows whether the matching image was found during preparation. |
    """)

    st.markdown("### Why the dataset and its splits matter")
    st.markdown(
        "Text, images, and labels let the project study both the post content and its visual context. "
        "The training split teaches the classifier; validation supports model selection; and the held-out "
        "test split gives a less biased estimate of performance. Keeping examples separated across these "
        "roles helps prevent the model from being evaluated on the same examples it learned from. "
        "This local subset is balanced between the two labels. Fakeddit labels use distant supervision, "
        "so they are research targets rather than claim-by-claim fact checks. A FAKE label or an image-text "
        "mismatch is not proof that a particular post is false."
    )

    st.markdown("### Dataset reference")
    st.markdown(
        "Fakeddit: Nakamura et al., *Fakeddit: A Multimodal Dataset for Fake News Classification* (LREC 2020). "
        "[Dataset website](https://fakeddit.netlify.app/) · "
        "[Project repository](https://github.com/entitize/Fakeddit)."
    )

if __name__ == "__main__":
    main()
