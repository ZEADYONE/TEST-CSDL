"""Evaluate HOG PCA + optional full PCA + balanced Linear SVM.

The experiment uses a deterministic per-species train/validation/test split.
PCA, scaling, and SVM are fit on train only during model selection. The chosen
configuration is then refit on train+validation and evaluated once on test.
Production retrieval is not modified by this script.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sqlalchemy import select
from sqlalchemy.orm import joinedload

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import ImageFeature, ImageRecord  # noqa: E402
from app.services.shape_features import SHAPE_DIM as GEOMETRY_DIM  # noqa: E402


HOG_DIMENSIONS = (64, 128, 256)
FULL_PCA_VARIANCES: tuple[float | None, ...] = (None, 0.90, 0.95, 0.98)
C_VALUES = (0.01, 0.1, 1.0, 10.0)
GRID_MAX_ITERATIONS = {0.01: 500, 0.1: 200, 1.0: 50, 10.0: 20}
OVERFIT_GAP = 0.20


@dataclass(frozen=True)
class ClassificationMetrics:
    top1: float
    top3: float
    top5: float
    macro_f1: float
    balanced_accuracy: float


def three_way_split(
    labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each species while ensuring every class appears in training."""

    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = len(indices)
        if count == 1:
            train.extend(indices.tolist())
            continue
        if count == 2:
            train.append(int(indices[0]))
            validation.append(int(indices[1]))
            continue
        test_count = max(1, int(round(0.20 * count)))
        validation_count = max(1, int(round(0.20 * count)))
        while test_count + validation_count >= count:
            if validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                break
        train_count = count - validation_count - test_count
        train.extend(indices[:train_count].tolist())
        validation.extend(indices[train_count : train_count + validation_count].tolist())
        test.extend(indices[train_count + validation_count :].tolist())
    return (
        np.asarray(sorted(train), dtype=np.int32),
        np.asarray(sorted(validation), dtype=np.int32),
        np.asarray(sorted(test), dtype=np.int32),
    )


def classification_metrics(
    labels: np.ndarray,
    decision: np.ndarray,
    classes: np.ndarray,
) -> ClassificationMetrics:
    order = np.argsort(-decision, axis=1)
    ranked_labels = classes[order]
    predictions = ranked_labels[:, 0]
    truth_classes = np.unique(labels)
    recalls = [
        float(np.mean(predictions[labels == label] == label))
        for label in truth_classes
    ]
    return ClassificationMetrics(
        top1=float(np.mean(predictions == labels)),
        top3=float(np.mean(np.any(ranked_labels[:, :3] == labels[:, None], axis=1))),
        top5=float(np.mean(np.any(ranked_labels[:, :5] == labels[:, None], axis=1))),
        macro_f1=float(
            f1_score(
                labels,
                predictions,
                labels=truth_classes,
                average="macro",
                zero_division=0,
            )
        ),
        balanced_accuracy=float(np.mean(recalls)),
    )


def metrics_dict(metrics: ClassificationMetrics) -> dict[str, float]:
    return {
        "top1": metrics.top1,
        "top3": metrics.top3,
        "top5": metrics.top5,
        "macro_f1": metrics.macro_f1,
        "balanced_accuracy": metrics.balanced_accuracy,
    }


def fit_hog_projection(
    hog: np.ndarray, fit_indices: np.ndarray, seed: int
) -> tuple[np.ndarray, PCA]:
    pca = PCA(
        n_components=max(HOG_DIMENSIONS),
        svd_solver="randomized",
        random_state=seed,
    ).fit(hog[fit_indices])
    return pca.transform(hog).astype(np.float32), pca


def standardized_representation(
    non_hog: np.ndarray,
    projected_hog: np.ndarray,
    hog_dimensions: int,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, StandardScaler]:
    combined = np.concatenate(
        (non_hog, projected_hog[:, :hog_dimensions]), axis=1
    ).astype(np.float32)
    scaler = StandardScaler().fit(combined[fit_indices])
    return scaler.transform(combined).astype(np.float32), scaler


def full_pca_representations(
    standardized: np.ndarray,
    fit_indices: np.ndarray,
) -> tuple[dict[float | None, np.ndarray], dict[str, dict[str, float | int]], PCA]:
    pca = PCA(n_components=max(v for v in FULL_PCA_VARIANCES if v is not None), svd_solver="full")
    pca.fit(standardized[fit_indices])
    projected = pca.transform(standardized).astype(np.float32)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    representations: dict[float | None, np.ndarray] = {None: standardized}
    details: dict[str, dict[str, float | int]] = {
        "none": {
            "dimensions": int(standardized.shape[1]),
            "explained_variance": 1.0,
        }
    }
    for target in (value for value in FULL_PCA_VARIANCES if value is not None):
        dimensions = min(int(np.searchsorted(cumulative, target) + 1), projected.shape[1])
        representations[target] = projected[:, :dimensions]
        details[f"{target:.2f}"] = {
            "dimensions": dimensions,
            "explained_variance": float(cumulative[dimensions - 1]),
        }
    return representations, details, pca


def fit_svm(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    c_value: float,
    seed: int,
    *,
    max_iter: int = 200,
    tolerance: float = 1e-2,
) -> tuple[LinearSVC, bool]:
    model = LinearSVC(
        C=c_value,
        class_weight="balanced",
        dual=True,
        tol=tolerance,
        max_iter=max_iter,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(features[train], labels[train])
    converged = int(model.n_iter_) < model.max_iter
    return model, converged


def evaluate_svm_config(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    c_value: float,
    seed: int,
) -> tuple[float, LinearSVC, bool, ClassificationMetrics, ClassificationMetrics]:
    model, converged = fit_svm(
        features,
        labels,
        train,
        c_value,
        seed,
        max_iter=GRID_MAX_ITERATIONS[c_value],
    )
    train_metrics = classification_metrics(
        labels[train], model.decision_function(features[train]), model.classes_
    )
    validation_metrics = classification_metrics(
        labels[validation],
        model.decision_function(features[validation]),
        model.classes_,
    )
    return c_value, model, converged, train_metrics, validation_metrics


def selection_key(payload: dict) -> tuple[int, float, float, float, float]:
    validation = payload["validation"]
    return (
        int(payload["converged"]),
        validation["top1"],
        validation["macro_f1"],
        validation["top3"],
        -payload["train_validation_gap"],
    )


def main() -> int:
    settings = get_settings()
    with SessionLocal() as session:
        records = list(
            session.scalars(
                select(ImageRecord)
                .options(joinedload(ImageRecord.feature))
                .join(ImageFeature)
                .where(
                    ImageRecord.species.is_not(None),
                    ImageFeature.feature_version == settings.feature_version,
                )
                .order_by(ImageRecord.id)
            ).all()
        )

    labels = np.asarray([str(record.species) for record in records], dtype=object)
    colors = np.asarray([record.feature.color_vector for record in records], dtype=np.float32)
    textures = np.asarray([record.feature.texture_vector for record in records], dtype=np.float32)
    shapes = np.asarray([record.feature.shape_vector for record in records], dtype=np.float32)
    non_hog = np.concatenate((colors, textures, shapes[:, :GEOMETRY_DIM]), axis=1)
    hog = shapes[:, GEOMETRY_DIM:]
    train, validation, test = three_way_split(labels, settings.random_seed)
    print(
        f"Feature version={settings.feature_version}, records={len(records)}, "
        f"species={len(set(labels.tolist()))}, train={len(train)}, "
        f"validation={len(validation)}, test={len(test)}"
    )
    print(
        f"Raw dimensions: non-HOG={non_hog.shape[1]}, HOG={hog.shape[1]}, "
        f"total={non_hog.shape[1] + hog.shape[1]}"
    )

    start = perf_counter()
    projected_hog, hog_pca = fit_hog_projection(hog, train, settings.random_seed)
    grid: dict[str, dict] = {}
    best_key: str | None = None
    best_payload: dict | None = None
    print(
        f"\n{'HOG':>5} {'Full PCA':>9} {'Dims':>6} {'C':>6} "
        f"{'Train':>8} {'Valid':>8} {'Gap':>8} {'Top3':>8} {'F1':>8}"
    )

    for hog_dimensions in HOG_DIMENSIONS:
        standardized, _ = standardized_representation(
            non_hog, projected_hog, hog_dimensions, train
        )
        representations, full_details, _ = full_pca_representations(
            standardized, train
        )
        for full_variance in FULL_PCA_VARIANCES:
            variant = representations[full_variance]
            detail_key = "none" if full_variance is None else f"{full_variance:.2f}"
            detail = full_details[detail_key]
            with ThreadPoolExecutor(max_workers=len(C_VALUES)) as executor:
                results = executor.map(
                    lambda c_value: evaluate_svm_config(
                        variant,
                        labels,
                        train,
                        validation,
                        c_value,
                        settings.random_seed,
                    ),
                    C_VALUES,
                )
                evaluated = list(results)
            for c_value, _model, converged, train_metrics, validation_metrics in evaluated:
                gap = train_metrics.top1 - validation_metrics.top1
                config_key = (
                    f"hog={hog_dimensions},full={detail_key},C={c_value:g}"
                )
                payload = {
                    "hog_dimensions": hog_dimensions,
                    "hog_explained_variance": float(
                        hog_pca.explained_variance_ratio_[:hog_dimensions].sum()
                    ),
                    "full_pca_target": full_variance,
                    "final_dimensions": int(detail["dimensions"]),
                    "full_pca_explained_variance": float(detail["explained_variance"]),
                    "C": c_value,
                    "class_weight": "balanced",
                    "converged": converged,
                    "train": metrics_dict(train_metrics),
                    "validation": metrics_dict(validation_metrics),
                    "train_validation_gap": gap,
                    "overfit_warning": gap > OVERFIT_GAP,
                }
                grid[config_key] = payload
                if best_payload is None or selection_key(payload) > selection_key(best_payload):
                    best_key = config_key
                    best_payload = payload

            representation_results = [
                item
                for item in grid.values()
                if item["hog_dimensions"] == hog_dimensions
                and item["full_pca_target"] == full_variance
            ]
            representation_best = max(representation_results, key=selection_key)
            print(
                f"{hog_dimensions:5d} {detail_key:>9} "
                f"{representation_best['final_dimensions']:6d} "
                f"{representation_best['C']:6g} "
                f"{representation_best['train']['top1']:8.4f} "
                f"{representation_best['validation']['top1']:8.4f} "
                f"{representation_best['train_validation_gap']:8.4f} "
                f"{representation_best['validation']['top3']:8.4f} "
                f"{representation_best['validation']['macro_f1']:8.4f}"
            )

    assert best_key is not None and best_payload is not None
    train_validation = np.sort(np.concatenate((train, validation)))
    selected_hog = int(best_payload["hog_dimensions"])
    selected_full = best_payload["full_pca_target"]
    selected_c = float(best_payload["C"])

    final_projected_hog, final_hog_pca = fit_hog_projection(
        hog, train_validation, settings.random_seed
    )
    final_standardized, _ = standardized_representation(
        non_hog, final_projected_hog, selected_hog, train_validation
    )
    final_representations, final_details, _ = full_pca_representations(
        final_standardized, train_validation
    )
    final_features = final_representations[selected_full]
    final_model, final_converged = fit_svm(
        final_features,
        labels,
        train_validation,
        selected_c,
        settings.random_seed,
        max_iter=20_000,
        tolerance=1e-4,
    )
    final_train_metrics = classification_metrics(
        labels[train_validation],
        final_model.decision_function(final_features[train_validation]),
        final_model.classes_,
    )
    test_metrics = classification_metrics(
        labels[test], final_model.decision_function(final_features[test]), final_model.classes_
    )
    final_gap = final_train_metrics.top1 - test_metrics.top1
    final_detail_key = "none" if selected_full is None else f"{selected_full:.2f}"

    print(f"\nSelected on validation: {best_key}")
    print(
        f"Refit train+validation: train top1={final_train_metrics.top1:.4f}, "
        f"test top1={test_metrics.top1:.4f}, gap={final_gap:.4f}"
    )
    print(
        f"Test: top1={test_metrics.top1:.4f}, top3={test_metrics.top3:.4f}, "
        f"top5={test_metrics.top5:.4f}, macro-F1={test_metrics.macro_f1:.4f}, "
        f"balanced={test_metrics.balanced_accuracy:.4f}"
    )

    report = {
        "feature_version": settings.feature_version,
        "protocol": (
            "stratified per-species train/validation/test; preprocessing fit on train; "
            "configuration selected on validation; refit on train+validation; test once"
        ),
        "split": {
            "seed": settings.random_seed,
            "records": len(records),
            "species": len(set(labels.tolist())),
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "train_species": len(set(labels[train].tolist())),
            "validation_species": len(set(labels[validation].tolist())),
            "test_species": len(set(labels[test].tolist())),
        },
        "search": {
            "hog_dimensions": HOG_DIMENSIONS,
            "full_pca_variances": ["none", 0.90, 0.95, 0.98],
            "C_values": C_VALUES,
            "class_weight": "balanced",
            "grid_max_iter_by_C": GRID_MAX_ITERATIONS,
            "grid_tolerance": 1e-2,
            "final_max_iter": 20_000,
            "final_tolerance": 1e-4,
            "selection_metric": "validation top1, macro-F1, top3, then smaller gap",
            "overfit_warning_threshold": OVERFIT_GAP,
        },
        "grid": grid,
        "selected_key": best_key,
        "selected_validation": best_payload,
        "final": {
            "hog_dimensions": selected_hog,
            "hog_explained_variance": float(
                final_hog_pca.explained_variance_ratio_[:selected_hog].sum()
            ),
            "full_pca_target": selected_full,
            "final_dimensions": int(final_details[final_detail_key]["dimensions"]),
            "full_pca_explained_variance": float(
                final_details[final_detail_key]["explained_variance"]
            ),
            "C": selected_c,
            "converged": final_converged,
            "train_validation_metrics": metrics_dict(final_train_metrics),
            "test_metrics": metrics_dict(test_metrics),
            "train_test_gap": final_gap,
            "overfit_warning": final_gap > OVERFIT_GAP,
        },
        "elapsed_seconds": perf_counter() - start,
    }
    report_path = settings.scaler_dir / "pca_svm_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
