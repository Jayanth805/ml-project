"""
TruthLens AI - Cross-Modal Consistency-Aware Multimodal Fake News Detection

A research-oriented system for detecting fake news by analyzing both text and images
using deep learning models including BERT, Vision Transformer (ViT), and CLIP.
"""

__version__ = "1.0.0"
__author__ = "TruthLens Research Team"

from .dataset import MultimodalFakeNewsDataset, Sample, FeatureCache, stratified_split
from .preprocessing import (
    TextPreprocessor,
    ImagePreprocessor,
    CLIPPreprocessor,
    MultimodalPreprocessor
)
from .encoders import BERTEncoder, ViTEncoder, CLIPEncoder, EncoderEnsemble, EncoderConfig
from .fusion import (
    ConsistencyAwareFusion,
    StandardFusion,
    TextOnlyClassifier,
    ImageOnlyClassifier,
    AblationFusion,
    FusionConfig,
    create_fusion_module
)
from .classifier import MultimodalClassifier, ClassifierConfig, create_classifier
from .train import Trainer, TrainingConfig, TrainingHistory
from .evaluate import Evaluator, evaluate_models, save_evaluation_results
from .predict import NewsAnalyzer, PredictionResult, EnsembleAnalyzer
from .metrics import (
    compute_metrics,
    compute_confusion_matrix,
    get_classification_report,
    compute_per_class_metrics,
    compare_models
)
from .plotting import (
    plot_training_history,
    plot_validation_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curves,
    plot_clip_similarity_distribution,
    plot_ablation_study,
    plot_image_swap_comparison
)
from .reporting import ReportGenerator, generate_experiment_summary
from .utils import (
    set_seed,
    load_config,
    save_config,
    get_device,
    print_gpu_info,
    ensure_dir,
    save_json,
    load_json,
    timestamp_now
)

__all__ = [
    # Dataset
    'MultimodalFakeNewsDataset',
    'Sample',
    'FeatureCache',
    'stratified_split',
    # Preprocessing
    'TextPreprocessor',
    'ImagePreprocessor',
    'CLIPPreprocessor',
    'MultimodalPreprocessor',
    # Encoders
    'BERTEncoder',
    'ViTEncoder',
    'CLIPEncoder',
    'EncoderEnsemble',
    'EncoderConfig',
    # Fusion
    'ConsistencyAwareFusion',
    'StandardFusion',
    'TextOnlyClassifier',
    'ImageOnlyClassifier',
    'AblationFusion',
    'FusionConfig',
    'create_fusion_module',
    # Classifier
    'MultimodalClassifier',
    'ClassifierConfig',
    'create_classifier',
    # Training
    'Trainer',
    'TrainingConfig',
    'TrainingHistory',
    # Evaluation
    'Evaluator',
    'evaluate_models',
    'save_evaluation_results',
    # Prediction
    'NewsAnalyzer',
    'PredictionResult',
    'EnsembleAnalyzer',
    # Metrics
    'compute_metrics',
    'compute_confusion_matrix',
    'get_classification_report',
    'compute_per_class_metrics',
    'compare_models',
    # Plotting
    'plot_training_history',
    'plot_validation_metrics',
    'plot_confusion_matrix',
    'plot_model_comparison',
    'plot_roc_curves',
    'plot_clip_similarity_distribution',
    'plot_ablation_study',
    'plot_image_swap_comparison',
    # Reporting
    'ReportGenerator',
    'generate_experiment_summary',
    # Utilities
    'set_seed',
    'load_config',
    'save_config',
    'get_device',
    'print_gpu_info',
    'ensure_dir',
    'save_json',
    'load_json',
    'timestamp_now',
]
