"""
TruthLens AI - Streamlit Web Application
Cross-Modal Consistency-Aware Multimodal Fake News Detection
"""

import os
import sys
import io
import json
import base64
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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-real {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.5rem;
        text-align: center;
    }
    .prediction-fake {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.5rem;
        text-align: center;
    }
    .disclaimer {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        margin-top: 1rem;
    }
    .sidebar-section {
        background-color: #f1f3f5;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Refined visual system shared by the app's research and analysis pages.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#eaf0ff; --muted:#a8b4d0; --line:rgba(179,198,232,.14); --mint:#7de3c3; }
    .stApp { background:radial-gradient(ellipse at 8% 0%,#1b2c4d 0,#0b1220 42%,#090f1b 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
    [data-testid="stHeader"] { background:rgba(9,15,27,.72); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#111b30,#0d1525); border-right:1px solid var(--line); }
    [data-testid="stSidebar"] label { color:var(--ink)!important; }
    h1,h2,h3,h4 { font-family:'Manrope',sans-serif!important; letter-spacing:-.035em; color:var(--ink)!important; }
    p,li { color:#c6d0e5; }
    [data-testid="stMetric"] { background:linear-gradient(145deg,rgba(28,43,70,.94),rgba(17,27,46,.88)); border:1px solid var(--line); border-radius:16px; padding:18px 20px; box-shadow:0 12px 28px rgba(0,0,0,.12); }
    [data-testid="stMetricLabel"] p { color:var(--muted)!important; font-size:.82rem!important; }
    [data-testid="stMetricValue"] { color:var(--ink)!important; font-family:'Manrope',sans-serif; }
    .stButton>button,.stDownloadButton>button { border-radius:11px; min-height:2.7rem; font-weight:700; border:1px solid var(--line); transition:all .18s ease; }
    .stButton>button[kind="primary"] { background:linear-gradient(100deg,#80e1c0,#79b9ff); color:#081321; border:0; }
    .stButton>button:hover,.stDownloadButton>button:hover { transform:translateY(-1px); border-color:var(--mint); }
    input,textarea,[data-baseweb="select"]>div { background:#101a2d!important; color:var(--ink)!important; border-color:var(--line)!important; border-radius:10px!important; }
    [data-testid="stExpander"] { background:rgba(17,27,46,.68); border:1px solid var(--line); border-radius:14px; }
    [data-testid="stFileUploader"] section { background:rgba(16,26,45,.72); border:1px dashed rgba(125,227,195,.4); border-radius:14px; }
    .hero { position:relative; overflow:hidden; padding:34px 38px; margin:8px 0 24px; border:1px solid rgba(171,198,245,.2); border-radius:24px; background:linear-gradient(115deg,rgba(35,58,95,.95),rgba(18,34,57,.9) 58%,rgba(22,57,67,.82)); box-shadow:0 24px 70px rgba(0,0,0,.24); }
    .hero:after { content:''; position:absolute; width:260px; height:260px; right:-68px; top:-125px; border-radius:50%; border:1px solid rgba(125,227,195,.28); box-shadow:0 0 0 26px rgba(125,227,195,.035),0 0 0 54px rgba(125,227,195,.025); }
    .eyebrow { color:var(--mint); font-size:.72rem; letter-spacing:.18em; text-transform:uppercase; font-weight:700; }
    .hero h1 { margin:10px 0 8px; font-size:2.55rem; line-height:1.12; }
    .hero p { max-width:720px; margin:0; color:#c2cee4; font-size:1rem; }
    .hero-meta { display:flex; gap:9px; flex-wrap:wrap; margin-top:22px; }
    .pill { display:inline-block; border:1px solid rgba(190,210,245,.2); background:rgba(5,13,25,.24); color:#dce8ff; border-radius:999px; padding:6px 11px; font-size:.76rem; }
    .section-label { margin:10px 0 14px; color:#dce7f8; font-family:'Manrope',sans-serif; font-weight:700; font-size:1.12rem; }
    .prediction-real { background:rgba(36,153,117,.2); color:#8ef0cd; border:1px solid rgba(125,227,195,.3); }
    .prediction-fake { background:rgba(200,74,98,.2); color:#ff9aa9; border:1px solid rgba(255,154,169,.3); }
    .disclaimer { background:rgba(206,160,58,.13); color:#f4d58a; border:1px solid rgba(230,187,89,.23); }
    .main-header { color:#eaf0ff; font-family:'Manrope',sans-serif; }
    .sub-header { color:#a8b4d0; }
    @media(max-width:700px) { .hero {padding:26px 22px;} .hero h1 {font-size:2rem;} }
</style>
""", unsafe_allow_html=True)

# Warm cream and terracotta editorial theme.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&display=swap');
    :root { --ink:#352a23; --muted:#79685a; --line:#e5d7c6; --terracotta:#a94f36; --terra-dark:#7e3829; --cream:#f5efe5; --paper:#fffaf2; --sage:#67745b; }
    .stApp { background:radial-gradient(ellipse at 5% 0%,#fffaf2 0,#f5efe5 53%,#eee3d4 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
    [data-testid="stHeader"] { background:rgba(245,239,229,.94); }
    [data-testid="stSidebar"] { display:none; }
    h1,h2,h3,h4 { font-family:'Fraunces',Georgia,serif!important; letter-spacing:-.025em; color:var(--ink)!important; }
    p,li { color:#57483b; }
    [data-testid="stMetric"] { background:rgba(255,250,242,.92); border:1px solid var(--line); border-radius:17px; padding:17px 19px; box-shadow:0 10px 26px rgba(89,60,37,.06); }
    [data-testid="stMetricLabel"] p { color:var(--muted)!important; font-size:.8rem!important; letter-spacing:.04em; text-transform:uppercase; }
    [data-testid="stMetricValue"] { color:var(--terra-dark)!important; font-family:'Fraunces',Georgia,serif; }
    .stButton>button,.stDownloadButton>button { border-radius:11px; min-height:2.65rem; font-weight:700; border:1px solid #d9c7b4; color:var(--ink); background:#fffaf2; transition:all .18s ease; }
    .stButton>button[kind="primary"] { background:var(--terracotta); color:#fffaf2; border:1px solid var(--terracotta); }
    .stButton>button:hover,.stDownloadButton>button:hover { transform:translateY(-1px); border-color:var(--terracotta); color:var(--terra-dark); }
    [data-testid="stRadio"] [role="radiogroup"] { display:flex; flex-direction:row; flex-wrap:wrap; gap:5px; padding:5px; border:1px solid #e4d5c4; border-radius:999px; background:#eee3d4; }
    [data-testid="stRadio"] label { padding:7px 14px; border-radius:999px; color:#604f41!important; font-size:.9rem; }
    [data-testid="stRadio"] label:has(input:checked) { background:var(--terracotta); color:#fffaf2!important; box-shadow:0 3px 9px rgba(126,56,41,.2); }
    [data-testid="stRadio"] label:has(input:checked) p { color:#fffaf2!important; }
    input,textarea,[data-baseweb="select"]>div { background:#fffaf2!important; color:var(--ink)!important; border-color:#decebc!important; border-radius:10px!important; }
    [data-testid="stExpander"] { background:rgba(255,250,242,.76); border:1px solid var(--line); border-radius:14px; }
    [data-testid="stFileUploader"] section { background:#fffaf2; border:1px dashed #c78c72; border-radius:14px; }
    .nav-brand { display:flex; align-items:center; gap:10px; min-height:48px; margin-bottom:4px; color:var(--ink); font-family:'Fraunces',Georgia,serif; font-size:1.23rem; font-weight:700; }
    .nav-mark { display:inline-grid; place-items:center; width:34px; height:34px; border-radius:50%; background:var(--terracotta); color:#fffaf2; font-family:'DM Sans',sans-serif; font-size:1rem; }
    .nav-caption { color:var(--muted); font-family:'DM Sans',sans-serif; font-size:.62rem; letter-spacing:.16em; text-transform:uppercase; }
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
def load_analyzer(checkpoint_path: str, device: str = "cuda"):
    """Load the analyzer with cached checkpoint."""
    try:
        analyzer = NewsAnalyzer(checkpoint_path=checkpoint_path)
        return analyzer
    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")
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
        'disclaimer': result.disclaimer
    }


def main():
    """Main application entry point."""
    init_session_state()

    pages = {
        "Analyze": "analysis",
        "Performance": "performance",
        "Experiments": "experiments",
        "Guide": "docs",
    }
    brand_col, links_col = st.columns([1.15, 3.4])
    with brand_col:
        st.markdown('<div class="nav-brand"><span class="nav-mark">T</span><span>TruthLens AI<br><span class="nav-caption">Research studio</span></span></div>', unsafe_allow_html=True)
    with links_col:
        selected_page = st.radio("Navigate", list(pages.keys()), horizontal=True, label_visibility="collapsed")

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
    elif pages[selected_page] == "experiments":
        show_experiments_page(config)
    elif pages[selected_page] == "docs":
        show_docs_page()


def show_analysis_page(config: Dict, default_checkpoint: Path):
    """News analysis page."""
    st.markdown('''<section class="hero"><div class="eyebrow">Multimodal intelligence / 01</div>
    <h1>TruthLens <span style="color:#a94f36">AI</span></h1>
    <p>Read the claim. Inspect the image. Surface signals that help you assess how well they fit together.</p>
    <div class="hero-meta"><span class="pill">BERT · language</span><span class="pill">ViT · vision</span><span class="pill">CLIP · alignment</span></div></section>''', unsafe_allow_html=True)

    summary_path = Path(__file__).parent / "data" / "processed" / "dataset_summary.json"
    try:
        dataset_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        dataset_summary = {}
    status_cols = st.columns(3)
    status_cols[0].metric("Model checkpoint", "READY" if default_checkpoint.is_file() else "PENDING")
    status_cols[1].metric("Prepared examples", f"{dataset_summary.get('total_samples', 0):,}")
    status_cols[2].metric("Compute device", "GPU" if torch.cuda.is_available() else "CPU")
    st.markdown("\n")

    # Model loading section
    with st.expander("⚙️ Model Settings", expanded=True):
        col1, col2 = st.columns([2, 1])

        checkpoint_path = col1.text_input(
            "Checkpoint Path",
            value=str(default_checkpoint) if default_checkpoint.exists() else "",
            help="Path to trained model checkpoint"
        )

        if col2.button("Load Model"):
            if checkpoint_path and os.path.exists(checkpoint_path):
                with st.spinner("Loading model..."):
                    st.session_state.analyzer = load_analyzer(checkpoint_path)
                    if st.session_state.analyzer:
                        st.session_state.checkpoint_loaded = st.session_state.analyzer.is_ready()
                        if st.session_state.checkpoint_loaded:
                            st.success("Model loaded successfully!")
                        else:
                            st.warning("Checkpoint structure may be incompatible.")
                    else:
                        st.session_state.checkpoint_loaded = False
            else:
                st.error("Checkpoint not found. Please train the model first or provide a valid path.")

        if st.session_state.checkpoint_loaded:
            st.info(f"✅ Model loaded: {st.session_state.analyzer.model_type}")
        else:
            st.warning("⚠️ No trained model loaded. You can still see CLIP similarity analysis.")

    st.markdown("---")

    # Input section
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📝 Input Text")
        text_input = st.text_area(
            "News Headline or Article",
            height=200,
            placeholder="Enter a news headline or article text here...",
            help="Enter the text content of the news you want to analyze"
        )

    with col2:
        st.markdown("### 🖼️ Image (Optional)")
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
    if analyze_clicked and text_input:
        if not text_input.strip():
            st.error("Please enter some text to analyze.")
            return

        with st.spinner("Analyzing..."):
            try:
                image_data = None
                if uploaded_file is not None:
                    image_data = Image.open(uploaded_file)

                if st.session_state.analyzer:
                    result = st.session_state.analyzer.analyze(text_input, image_data)
                else:
                    # Create analyzer without checkpoint for CLIP analysis
                    analyzer = NewsAnalyzer()
                    result = analyzer.analyze(text_input, image_data)

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

        # Prediction display
        pred_class = result['prediction']
        if pred_class == "FAKE":
            st.markdown(f'<div class="prediction-fake">🚨 FAKE NEWS DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-real">✅ APPEARS LEGITIMATE</div>', unsafe_allow_html=True)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Probability REAL",
                f"{result['probability_real']:.1%}",
                help="Estimated probability that this is legitimate news"
            )
        with col2:
            st.metric(
                "Probability FAKE",
                f"{result['probability_fake']:.1%}",
                help="Estimated probability that this is fake news"
            )
        with col3:
            st.metric(
                "Confidence",
                f"{result['confidence']:.1%}",
                help="Overall confidence in the prediction"
            )
        with col4:
            st.metric(
                "CLIP Similarity",
                f"{result['clip_similarity']:.3f}",
                help="Semantic similarity between text and image"
            )

        # Consistency interpretation
        st.markdown("#### 🔗 Image-Text Consistency")
        interp = result['consistency_interpretation']

        consistency_col1, consistency_col2 = st.columns(2)
        with consistency_col1:
            st.markdown(f"**Interpretation:** {interp.get('interpretation', 'N/A')}")
            st.markdown(f"**Level:** {interp.get('level', 'N/A').replace('-', ' ').title()}")

        with consistency_col2:
            if interp.get('warning'):
                st.markdown(f"⚠️ {interp['warning']}")

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
            if st.button("📝 View Text Report", use_container_width=True):
                with st.expander("Report Content", expanded=True):
                    st.text(f"TruthLens AI - Prediction Report")
                    st.text(f"Generated: {result['timestamp']}")
                    st.text("-" * 40)
                    st.text(f"Prediction: {result['prediction']}")
                    st.text(f"Probability REAL: {result['probability_real']:.2%}")
                    st.text(f"Probability FAKE: {result['probability_fake']:.2%}")
                    st.text(f"CLIP Similarity: {result['clip_similarity']:.4f}")
                    st.text("-" * 40)
                    st.text("Disclaimer:")
                    st.text(result['disclaimer'])

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


def show_experiments_page(config: Dict):
    """Research experiments page."""
    st.markdown('<p class="main-header">Research Experiments</p>', unsafe_allow_html=True)

    st.markdown("""
    This page presents the research experiments conducted to evaluate the effectiveness
    of the consistency-aware multimodal fusion approach for fake news detection.
    """)

    # Image-swap experiment
    st.markdown("### 🔄 Image-Swap Experiment")

    st.markdown("""
    The image-swap experiment tests model robustness by:
    1. Keeping original text and news-truth labels
    2. Replacing images with unrelated images
    3. Comparing predictions on original vs. swapped pairs
    """)

    # Check for experiment results
    exp_dir = Path(__file__).parent / "results"
    swap_results_file = exp_dir / "image_swap_results.json"

    if swap_results_file.exists():
        try:
            swap_results = json.load(open(swap_results_file))

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Original Pairs")
                st.json(swap_results.get('original', {}))

            with col2:
                st.markdown("#### Image-Swapped Pairs")
                st.json(swap_results.get('swapped', {}))

            st.markdown("#### Performance Difference")
            st.json(swap_results.get('metric_differences', {}))

        except Exception as e:
            st.error(f"Error loading results: {e}")
    else:
        st.info("No image-swap experiment results found. Run the experiment notebooks to generate results.")

    # Ablation study
    st.markdown("---")
    st.markdown("### 🔬 Ablation Study")

    st.markdown("""
    The ablation study compares:
    - **Model A**: Text-only (BERT)
    - **Model B**: Image-only (ViT)
    - **Model C**: Standard multimodal fusion
    - **Model D**: Consistency-aware fusion (proposed)
    - **Model E**: Ablation without CLIP similarity
    """)

    # Check for ablation results
    ablation_results_file = exp_dir / "ablation_results.json"

    if ablation_results_file.exists():
        try:
            ablation = json.load(open(ablation_results_file))

            import pandas as pd
            df = pd.DataFrame(ablation)
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading ablation results: {e}")
    else:
        st.info("No ablation study results found.")

    # Limitations
    st.markdown("---")
    st.markdown("### ⚠️ Important Notes")

    st.markdown("""
    1. **Dataset Labels**: The Fakeddit dataset uses distant supervision labels, which may contain noise.
       Not all examples labeled as "FAKE" are necessarily false, and vice versa.

    2. **Semantic vs. Factual**: CLIP similarity measures semantic alignment between image and text,
       NOT factual accuracy of the news content.

    3. **Domain Shift**: Models trained on Reddit data may not generalize well to other news sources
       with different writing styles, image types, or topics.

    4. **Image Swap ≠ Fake**: Replacing an image with an unrelated one does NOT automatically make
       the news fake. The model's response to such swaps reveals its reliance on visual-textual
       consistency rather than verifying truthfulness.
    """)


def show_docs_page():
    """Documentation page."""
    st.markdown('<p class="main-header">Documentation</p>', unsafe_allow_html=True)

    docs_dir = Path(__file__).parent / "docs"

    # Project overview
    with st.expander("📋 Project Overview", expanded=True):
        st.markdown("""
        **TruthLens AI** is a research-oriented system for detecting fake news by analyzing
        both text and images using deep learning.

        ### Architecture

        The system uses three main components:

        1. **Text Encoder (BERT)**: Extracts semantic features from news text
        2. **Image Encoder (ViT)**: Extracts visual features from images
        3. **CLIP Model**: Measures semantic similarity between text and images

        ### Research Question

        Does explicitly incorporating image-text semantic consistency improve multimodal
        fake-news classification compared with standard text-image feature fusion?
        """)

    # Model details
    with st.expander("🧠 Model Architecture"):
        st.markdown("""
        ### Multimodal Fusion Strategy

        The proposed consistency-aware fusion combines:

        - **Projected text features** (768 → 256 dimensions)
        - **Projected image features** (768 → 256 dimensions)
        - **Absolute difference** |T_proj - I_proj|
        - **Element-wise product** T_proj ⊙ I_proj
        - **CLIP similarity score**

        ### Fusion Vector Dimension

        ```
        256 (text) + 256 (image) + 256 (diff) + 256 (product) + 1 (similarity) = 1025
        ```

        ### Classification Head

        ```
        Linear(1025, 256) → ReLU → Dropout(0.3) → Linear(256, 1)
        ```
        """)

    # Dataset info
    with st.expander("📁 Dataset Information"):
        st.markdown("""
        ### Primary Dataset: Fakeddit

        The Fakeddit dataset is a multimodal fake news dataset collected from Reddit.

        **Citation:**
        ```
        @inproceedings{nakamura2020fakeddit,
          title={Fakeddit: A Multimodal Dataset for Fake News Classification},
          author={Nakamura, Kai and others},
          booktitle={LREC},
          year={2020}
        }
        ```

        ### Dataset Access

        Visit [https://fakeddit.netlify.app/](https://fakeddit.netlify.app/) for dataset access.

        ### Label Information

        The dataset uses 3-way labels:
        - 0: Reliable
        - 1: Unreliable
        - 2: Misleading

        For binary classification, we map:
        - Reliable → 0 (REAL)
        - Unreliable + Misleading → 1 (FAKE)
        """)

    # Training guide
    with st.expander("🚀 Training Guide"):
        st.markdown("""
        ### Local Training

        ```bash
        # Install dependencies
        pip install -r requirements.txt

        # Train consistency-aware model
        python scripts/train_model.py --config configs/experiment.yaml --model consistency_aware

        # Evaluate model
        python scripts/evaluate_model.py --checkpoint artifacts/checkpoints/consistency_aware_best.pt
        ```

        ### Google Colab Training

        1. Open `notebooks/04_consistency_aware_training.ipynb` in Google Colab
        2. Upload your dataset to Google Drive
        3. Update dataset paths in the notebook
        4. Run cells sequentially with GPU runtime

        ### Training Configuration

        Key parameters in `configs/experiment.yaml`:
        - `training.batch_size`: 16 (adjust for GPU memory)
        - `training.learning_rate`: 1e-3
        - `training.epochs`: 5-10
        - `model.freeze_encoders`: true (keep encoders frozen)
        """)

    # Limitations
    with st.expander("⚠️ Limitations & Disclaimer"):
        st.markdown("""
        ### Known Limitations

        1. **Distant Supervision Labels**: Fakeddit uses automatic labeling based on
           subreddit membership, which may introduce noise.

        2. **Semantic ≠ Factual**: CLIP measures whether an image semantically matches
           the text, not whether the content is factually accurate.

        3. **Domain Specificity**: Trained on Reddit posts, may not generalize to
           formal news articles or other domains.

        4. **Image Quality**: Corrupt or low-quality images affect performance.

        5. **Text Length**: Limited to 128 tokens for BERT processing.

        ### Important Disclaimer

        **This system is for research purposes only.**

        - Predictions are based on learned patterns, not verified facts
        - Semantic similarity does not prove news truthfulness
        - Always verify information from authoritative sources
        - Do not use this system for making critical decisions
        """)

    # References
    with st.expander("📚 References"):
        st.markdown("""
        ### Papers

        1. **BERT**: Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers"
        2. **ViT**: Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition"
        3. **CLIP**: Radford et al. "Learning Transferable Visual Models From Natural Language Supervision"
        4. **Fakeddit**: Nakamura et al. "Fakeddit: A Multimodal Dataset for Fake News Classification"
        5. **NewsCLIPpings**: Luo et al. "NewsCLIPpings: A Dataset for Out-of-Context News Detection"

        ### Libraries

        - PyTorch: https://pytorch.org/
        - Hugging Face Transformers: https://huggingface.co/transformers/
        - Streamlit: https://streamlit.io/
        """)


if __name__ == "__main__":
    main()
