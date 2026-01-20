import pytest
import sys
import os

# Add src path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)

from src.pipeline_steps_df.model_promotion_decision import should_promote_model

def test_promote_better_candidate():
    """Candidate is better on all metrics."""
    baseline = {"accuracy": 0.8, "weighted_f1": 0.8, "macro_f1": 0.8, "mean_confidence": 0.8}
    candidate = {"accuracy": 0.85, "weighted_f1": 0.85, "macro_f1": 0.85, "mean_confidence": 0.85}
    assert should_promote_model(baseline, candidate) is True

def test_reject_worse_f1():
    """Candidate F1 drops significantly."""
    baseline = {"accuracy": 0.8, "weighted_f1": 0.8, "macro_f1": 0.8, "mean_confidence": 0.8}
    candidate = {"accuracy": 0.8, "weighted_f1": 0.75, "macro_f1": 0.8, "mean_confidence": 0.8} # Drop 0.05 > 0.001
    assert should_promote_model(baseline, candidate) is False

def test_reject_worse_accuracy():
    """Candidate Accuracy drops significantly."""
    baseline = {"accuracy": 0.8, "weighted_f1": 0.8, "macro_f1": 0.8, "mean_confidence": 0.8}
    candidate = {"accuracy": 0.79, "weighted_f1": 0.8, "macro_f1": 0.8, "mean_confidence": 0.8} # Drop 0.01 > 0.002
    assert should_promote_model(baseline, candidate) is False

def test_allow_slight_drop():
    """Candidate drops slightly but within tolerance."""
    baseline = {"accuracy": 0.8, "weighted_f1": 0.8000, "macro_f1": 0.8, "mean_confidence": 0.8}
    candidate = {"accuracy": 0.7999, "weighted_f1": 0.79995, "macro_f1": 0.8, "mean_confidence": 0.8} 
    # Drops are negligible
    assert should_promote_model(baseline, candidate) is True

def test_reject_macro_f1_regression():
    """Strict check on macro_f1 (no regression allowed)."""
    baseline = {"accuracy": 0.8, "weighted_f1": 0.8, "macro_f1": 0.80, "mean_confidence": 0.8}
    candidate = {"accuracy": 0.8, "weighted_f1": 0.8, "macro_f1": 0.79, "mean_confidence": 0.8}
    assert should_promote_model(baseline, candidate) is False
