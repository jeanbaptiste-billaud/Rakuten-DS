"""
mlflow_model_promotion.py

Décision de promotion d'un modèle de classification à partir des métriques déjà loggées dans MLflow.
- Compare baseline (modèle en service) vs candidate (nouveau run)
- Verdict: PROMOTE | REJECT | REVIEW
- Basé sur: accuracy, weighted_f1, macro_f1, mean_confidence (noms configurables)

Pré-requis:
pip install mlflow

Usage:
python mlflow_model_promotion.py --mlflow-uri http://localhost:5000 \
  --baseline-run-id <RUN_ID_BASELINE> --candidate-run-id <RUN_ID_CANDIDATE>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient


# ----------------------------
# Config / Policy
# ----------------------------

@dataclass(frozen=True)
class MetricKeys:
    accuracy: str = "accuracy"
    weighted_f1: str = "weighted_f1"
    macro_f1: str = "macro_f1"
    mean_confidence: str = "mean_confidence"


@dataclass(frozen=True)
class PromotionPolicy:
    # Tolérances (scores en [0,1])
    max_drop_weighted_f1: float = 0.001   # baisse max autorisée
    max_drop_accuracy: float = 0.002      # baisse max autorisée
    require_macro_f1_not_worse: bool = True

    # Mean confidence: signal, pas hard gate
    confidence_drop_warn: float = 0.02    # si baisse > warn => REVIEW (si tout le reste OK)

    # Si une métrique manque: on décide quoi ?
    missing_metric_verdict: str = "REVIEW"  # "REVIEW" ou "REJECT"


@dataclass
class DecisionResult:
    verdict: str
    reasons: List[str]
    deltas: Dict[str, Optional[float]]
    baseline_metrics: Dict[str, Any]
    candidate_metrics: Dict[str, Any]
    policy: Dict[str, Any]
    metric_keys: Dict[str, Any]


# ----------------------------
# Helpers
# ----------------------------

def _round4(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 4)


def get_run_metrics(client: MlflowClient, run_id: str) -> Dict[str, Any]:
    run = client.get_run(run_id)
    # run.data.metrics: Dict[str, float]
    # run.data.params: Dict[str, str]
    # run.data.tags: Dict[str, str]
    return {
        "run_id": run_id,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
        "tags": dict(run.data.tags),
        "artifact_uri": run.info.artifact_uri,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
    }


def extract_required_metrics(
    run_payload: Dict[str, Any],
    keys: MetricKeys,
) -> Dict[str, Optional[float]]:
    m = run_payload["metrics"]
    out = {
        "accuracy": m.get(keys.accuracy),
        "weighted_f1": m.get(keys.weighted_f1),
        "macro_f1": m.get(keys.macro_f1),
        "mean_confidence": m.get(keys.mean_confidence),
    }
    # cast to float if present
    for k, v in list(out.items()):
        out[k] = None if v is None else float(v)
    return out


def compare_and_decide(
    baseline: Dict[str, Optional[float]],
    candidate: Dict[str, Optional[float]],
    policy: PromotionPolicy,
) -> DecisionResult:
    reasons: List[str] = []
    verdict = "PROMOTE"

    def missing(name: str) -> bool:
        return baseline.get(name) is None or candidate.get(name) is None

    # Compute deltas (candidate - baseline)
    deltas = {
        "accuracy": None if missing("accuracy") else _round4(candidate["accuracy"] - baseline["accuracy"]),
        "weighted_f1": None if missing("weighted_f1") else _round4(candidate["weighted_f1"] - baseline["weighted_f1"]),
        "macro_f1": None if missing("macro_f1") else _round4(candidate["macro_f1"] - baseline["macro_f1"]),
        "mean_confidence": None if missing("mean_confidence") else _round4(candidate["mean_confidence"] - baseline["mean_confidence"]),
    }

    # Missing metrics handling
    required_for_gating = ["accuracy", "weighted_f1", "macro_f1"]
    missing_required = [k for k in required_for_gating if missing(k)]
    if missing_required:
        verdict = policy.missing_metric_verdict
        reasons.append(f"Métriques manquantes pour décider proprement: {missing_required}.")
        # On n'arrête pas forcément: on peut encore signaler d'autres infos si dispo.

    # Hard gates
    if not missing("weighted_f1"):
        drop = baseline["weighted_f1"] - candidate["weighted_f1"]
        if drop > policy.max_drop_weighted_f1:
            verdict = "REJECT"
            reasons.append(
                f"Weighted F1 baisse trop: {baseline['weighted_f1']:.4f} -> {candidate['weighted_f1']:.4f} "
                f"(drop={drop:.4f} > max={policy.max_drop_weighted_f1:.4f})."
            )

    if not missing("accuracy"):
        drop = baseline["accuracy"] - candidate["accuracy"]
        if drop > policy.max_drop_accuracy:
            verdict = "REJECT"
            reasons.append(
                f"Accuracy baisse trop: {baseline['accuracy']:.4f} -> {candidate['accuracy']:.4f} "
                f"(drop={drop:.4f} > max={policy.max_drop_accuracy:.4f})."
            )

    if policy.require_macro_f1_not_worse and not missing("macro_f1"):
        if candidate["macro_f1"] < baseline["macro_f1"]:
            verdict = "REJECT"
            reasons.append(
                f"Macro F1 régresse: {baseline['macro_f1']:.4f} -> {candidate['macro_f1']:.4f}."
            )

    # Soft warning: confidence drop
    if verdict == "PROMOTE" and not missing("mean_confidence"):
        drop = baseline["mean_confidence"] - candidate["mean_confidence"]
        if drop > policy.confidence_drop_warn:
            verdict = "REVIEW"
            reasons.append(
                f"Mean confidence baisse notable: {baseline['mean_confidence']:.4f} -> {candidate['mean_confidence']:.4f} "
                f"(drop={drop:.4f} > warn={policy.confidence_drop_warn:.4f})."
            )

    if not reasons:
        reasons.append("OK: aucune règle de régression déclenchée.")

    return DecisionResult(
        verdict=verdict,
        reasons=reasons,
        deltas=deltas,
        baseline_metrics={k: _round4(v) for k, v in baseline.items()},
        candidate_metrics={k: _round4(v) for k, v in candidate.items()},
        policy=asdict(policy),
        metric_keys={},  # rempli au niveau main
    )


# ----------------------------
# Optional: récupérer le run "en service" via Model Registry
# ----------------------------

def get_run_id_from_registry(
    client: MlflowClient,
    model_name: str,
    stage: str = "Production",
) -> str:
    """
    Retourne le run_id du modèle dans un stage (ex: Production).
    Nécessite que vous utilisiez le Model Registry MLflow.
    """
    mv = client.get_latest_versions(model_name, stages=[stage])
    if not mv:
        raise RuntimeError(f"Aucune version trouvée pour le modèle '{model_name}' au stage '{stage}'.")
    # On prend la 1ère
    source = mv[0].source  # chemin de l'artefact
    run_id = mv[0].run_id  # MLflow model version stores run_id
    if not run_id:
        raise RuntimeError(f"Impossible de récupérer run_id depuis le registry pour '{model_name}' ({stage}).")
    return run_id


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-uri", default=None, help="Tracking URI (ex: http://localhost:5000)")
    parser.add_argument("--baseline-run-id", default=None, help="Run ID du modèle en service (baseline)")
    parser.add_argument("--candidate-run-id", required=True, help="Run ID du nouveau modèle (candidate)")
    parser.add_argument("--registry-model-name", default=None, help="Optionnel: nom du modèle dans MLflow Registry")
    parser.add_argument("--registry-stage", default="Production", help="Stage du modèle baseline (default: Production)")
    parser.add_argument("--out-json", default="promotion_decision.json", help="Chemin de sortie JSON")

    # Policy overrides rapides
    parser.add_argument("--max-drop-weighted-f1", type=float, default=0.001)
    parser.add_argument("--max-drop-accuracy", type=float, default=0.002)
    parser.add_argument("--macro-f1-not-worse", action="store_true", default=True)
    parser.add_argument("--confidence-drop-warn", type=float, default=0.02)
    parser.add_argument("--missing-metric-verdict", choices=["REVIEW", "REJECT"], default="REVIEW")

    # Metric keys (si vos noms changent dans MLflow)
    parser.add_argument("--key-accuracy", default="accuracy")
    parser.add_argument("--key-weighted-f1", default="weighted_f1")
    parser.add_argument("--key-macro-f1", default="macro_f1")
    parser.add_argument("--key-mean-confidence", default="mean_confidence")

    args = parser.parse_args()

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)

    client = MlflowClient()

    keys = MetricKeys(
        accuracy=args.key_accuracy,
        weighted_f1=args.key_weighted_f1,
        macro_f1=args.key_macro_f1,
        mean_confidence=args.key_mean_confidence,
    )

    policy = PromotionPolicy(
        max_drop_weighted_f1=args.max_drop_weighted_f1,
        max_drop_accuracy=args.max_drop_accuracy,
        require_macro_f1_not_worse=args.macro_f1_not_worse,
        confidence_drop_warn=args.confidence_drop_warn,
        missing_metric_verdict=args.missing_metric_verdict,
    )

    # Baseline run id: direct ou via registry
    baseline_run_id = args.baseline_run_id
    if baseline_run_id is None:
        if not args.registry_model_name:
            raise SystemExit(
                "Il faut fournir --baseline-run-id OU (--registry-model-name pour récupérer la baseline via le registry)."
            )
        baseline_run_id = get_run_id_from_registry(client, args.registry_model_name, args.registry_stage)

    # Fetch runs
    baseline_payload = get_run_metrics(client, baseline_run_id)
    candidate_payload = get_run_metrics(client, args.candidate_run_id)

    baseline_metrics = extract_required_metrics(baseline_payload, keys)
    candidate_metrics = extract_required_metrics(candidate_payload, keys)

    decision = compare_and_decide(baseline_metrics, candidate_metrics, policy)
    decision.metric_keys = asdict(keys)

    # Enrich JSON avec infos utiles
    output = asdict(decision)
    output["baseline_run_info"] = {
        "run_id": baseline_run_id,
        "artifact_uri": baseline_payload["artifact_uri"],
        "status": baseline_payload["status"],
    }
    output["candidate_run_info"] = {
        "run_id": args.candidate_run_id,
        "artifact_uri": candidate_payload["artifact_uri"],
        "status": candidate_payload["status"],
    }

    # Print
    print("\n=== VERDICT ===")
    print(output["verdict"])
    print("\n=== RAISONS ===")
    for r in output["reasons"]:
        print(f"- {r}")
    print("\n=== METRICS (baseline -> candidate) ===")
    for k in ["accuracy", "weighted_f1", "macro_f1", "mean_confidence"]:
        b = output["baseline_metrics"].get(k)
        c = output["candidate_metrics"].get(k)
        d = output["deltas"].get(k)
        print(f"- {k}: {b} -> {c} (delta={d})")

    # Save JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ JSON écrit: {args.out_json}")


if __name__ == "__main__":
    main()
