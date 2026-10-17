"""Evaluate PCA + Linear SVM species scores as a soft visual reranker.

SVM/PCA hyperparameters come from Method 7. Alpha is selected with the
validation split using an SVM fit on train only. The chosen policies are then
evaluated on test with an SVM refit on train+validation. No hard species filter
is used.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import joinedload

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import ImageFeature, ImageRecord  # noqa: E402
from app.services.shape_features import SHAPE_DIM as GEOMETRY_DIM  # noqa: E402
from scripts.evaluate_knn_soft_reranking import (  # noqa: E402
    normalize_species_scores,
    soft_reranked_distance,
)
from scripts.evaluate_pca_svm import (  # noqa: E402
    fit_hog_projection,
    fit_svm,
    full_pca_representations,
    standardized_representation,
    three_way_split,
)
from scripts.evaluate_species_prototypes import (  # noqa: E402
    group_raw_distances,
    normalize_and_fuse,
)
from scripts.evaluate_weighted_knn import weighted_species_scores  # noqa: E402
from scripts.tune_weights import evaluate, robust_normalize  # noqa: E402


ALPHAS = (0.9, 0.8, 0.7, 0.6)
KNN_REFERENCE_ALPHAS = (0.9, 0.5)


def build_svm_features(
    non_hog: np.ndarray,
    hog: np.ndarray,
    fit_indices: np.ndarray,
    *,
    hog_dimensions: int,
    full_pca_target: float | None,
    seed: int,
) -> np.ndarray:
    projected_hog, _ = fit_hog_projection(hog, fit_indices, seed)
    standardized, _ = standardized_representation(
        non_hog, projected_hog, hog_dimensions, fit_indices
    )
    if full_pca_target is None:
        return standardized
    representations, _, _ = full_pca_representations(standardized, fit_indices)
    return representations[full_pca_target]


def normalized_svm_scores(model, features: np.ndarray, species: np.ndarray) -> np.ndarray:
    """Map uncalibrated SVM margins to per-query [0, 1] species evidence."""

    decision = np.asarray(model.decision_function(features), dtype=np.float32)
    minimum = decision.min(axis=1, keepdims=True)
    maximum = decision.max(axis=1, keepdims=True)
    normalized = (decision - minimum) / np.maximum(maximum - minimum, 1e-10)
    if np.array_equal(model.classes_, species):
        return normalized.astype(np.float32)
    output = np.zeros((len(features), len(species)), dtype=np.float32)
    species_to_index = {label: index for index, label in enumerate(species)}
    for class_index, label in enumerate(model.classes_):
        output[:, species_to_index[str(label)]] = normalized[:, class_index]
    return output


def select_policies(calibration_grid: dict[str, dict]) -> tuple[float, float]:
    map_alpha = max(
        ALPHAS,
        key=lambda alpha: metric_key_from_dict(calibration_grid[f"{alpha:.1f}"]),
    )
    p1_alpha = max(
        ALPHAS,
        key=lambda alpha: (
            calibration_grid[f"{alpha:.1f}"]["p1"],
            calibration_grid[f"{alpha:.1f}"]["map10"],
            calibration_grid[f"{alpha:.1f}"]["p5"],
        ),
    )
    return map_alpha, p1_alpha


def metric_key_from_dict(metrics: dict[str, float]) -> tuple[float, float, float]:
    return metrics["map10"], metrics["p1"], metrics["p5"]


def main() -> int:
    settings = get_settings()
    method7_path = settings.scaler_dir / "pca_svm_report.json"
    if not method7_path.is_file():
        raise FileNotFoundError(
            "Method 7 report is missing. Run scripts/evaluate_pca_svm.py first."
        )
    method7 = json.loads(method7_path.read_text(encoding="utf-8"))
    selected = method7["final"]
    hog_dimensions = int(selected["hog_dimensions"])
    full_pca_target = selected["full_pca_target"]
    c_value = float(selected["C"])

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
    species = np.asarray(sorted(set(labels.tolist())), dtype=object)
    species_to_index = {name: index for index, name in enumerate(species)}
    label_indices = np.asarray([species_to_index[name] for name in labels], dtype=np.int32)
    species_counts = np.bincount(label_indices, minlength=len(species)).astype(np.float32)
    colors = np.asarray([record.feature.color_vector for record in records], dtype=np.float32)
    textures = np.asarray([record.feature.texture_vector for record in records], dtype=np.float32)
    shapes = np.asarray([record.feature.shape_vector for record in records], dtype=np.float32)
    feature_groups = colors, textures, shapes
    non_hog = np.concatenate((colors, textures, shapes[:, :GEOMETRY_DIM]), axis=1)
    hog = shapes[:, GEOMETRY_DIM:]
    train, validation, test = three_way_split(labels, settings.random_seed)
    print(
        f"Feature version={settings.feature_version}, records={len(records)}, "
        f"species={len(species)}, train={len(train)}, validation={len(validation)}, "
        f"test={len(test)}"
    )
    print(
        f"Method 7 configuration: HOG={hog_dimensions}, full PCA={full_pca_target}, "
        f"C={c_value:g}"
    )

    print("Computing visual distances...")
    visual_groups = group_raw_distances(feature_groups, feature_groups, settings)
    visual_ranges = [robust_normalize(group)[1]["p95"] for group in visual_groups]
    visual_distance = normalize_and_fuse(visual_groups, visual_ranges, settings)
    np.fill_diagonal(visual_distance, np.inf)

    print("Fitting train-only PCA/SVM for alpha selection...")
    validation_features = build_svm_features(
        non_hog,
        hog,
        train,
        hog_dimensions=hog_dimensions,
        full_pca_target=full_pca_target,
        seed=settings.random_seed,
    )
    validation_model, validation_converged = fit_svm(
        validation_features,
        labels,
        train,
        c_value,
        settings.random_seed,
        max_iter=20_000,
        tolerance=1e-4,
    )
    validation_scores = normalized_svm_scores(
        validation_model, validation_features, species
    )

    calibration_grid: dict[str, dict] = {}
    print(
        f"\n{'alpha':>7} {'visual':>8} {'SVM':>8} {'P@1':>8} {'P@5':>8} "
        f"{'P@10':>8} {'R@10':>8} {'mAP@10':>8}"
    )
    for alpha in ALPHAS:
        distance = soft_reranked_distance(
            visual_distance, validation_scores, label_indices, alpha
        )
        metrics = evaluate(distance, labels, validation)
        calibration_grid[f"{alpha:.1f}"] = asdict(metrics)
        print(
            f"{alpha:7.1f} {alpha:8.1f} {1-alpha:8.1f} "
            f"{metrics.p1:8.4f} {metrics.p5:8.4f} {metrics.p10:8.4f} "
            f"{metrics.recall10:8.4f} {metrics.map10:8.4f}"
        )
    map_alpha, p1_alpha = select_policies(calibration_grid)

    print("\nRefitting PCA/SVM on train+validation for held-out test...")
    train_validation = np.sort(np.concatenate((train, validation)))
    final_features = build_svm_features(
        non_hog,
        hog,
        train_validation,
        hog_dimensions=hog_dimensions,
        full_pca_target=full_pca_target,
        seed=settings.random_seed,
    )
    fit_start = perf_counter()
    final_model, final_converged = fit_svm(
        final_features,
        labels,
        train_validation,
        c_value,
        settings.random_seed,
        max_iter=20_000,
        tolerance=1e-4,
    )
    fit_seconds = perf_counter() - fit_start
    score_start = perf_counter()
    test_scores_only = normalized_svm_scores(
        final_model, final_features[test], species
    )
    score_seconds_per_query = (perf_counter() - score_start) / len(test)
    final_scores = np.zeros((len(records), len(species)), dtype=np.float32)
    final_scores[test] = test_scores_only

    baseline_test = evaluate(visual_distance, labels, test)
    tested_alphas = sorted(set((map_alpha, p1_alpha)), reverse=True)
    svm_test: dict[str, dict] = {}
    for alpha in tested_alphas:
        metrics = evaluate(
            soft_reranked_distance(
                visual_distance, final_scores, label_indices, alpha
            ),
            labels,
            test,
        )
        svm_test[f"{alpha:.1f}"] = asdict(metrics)

    knn_raw = weighted_species_scores(
        visual_distance,
        label_indices,
        species_counts,
        neighbors=10,
        power=1,
        beta=0.0,
    )
    knn_scores = normalize_species_scores(knn_raw)
    knn_reference: dict[str, dict] = {}
    for alpha in KNN_REFERENCE_ALPHAS:
        metrics = evaluate(
            soft_reranked_distance(
                visual_distance, knn_scores, label_indices, alpha
            ),
            labels,
            test,
        )
        knn_reference[f"{alpha:.1f}"] = asdict(metrics)

    print(
        f"Selected on validation: alpha={p1_alpha:.1f} for P@1; "
        f"alpha={map_alpha:.1f} for mAP@10"
    )
    print(f"\n{'Test system':<28} {'P@1':>8} {'P@5':>8} {'P@10':>8} {'R@10':>8} {'mAP@10':>8}")
    systems = {"Visual baseline": asdict(baseline_test)}
    systems.update({f"SVM soft alpha={key}": value for key, value in svm_test.items()})
    systems.update({f"kNN soft alpha={key}": value for key, value in knn_reference.items()})
    for name, metrics in systems.items():
        print(
            f"{name:<28} {metrics['p1']:8.4f} {metrics['p5']:8.4f} "
            f"{metrics['p10']:8.4f} {metrics['recall10']:8.4f} "
            f"{metrics['map10']:8.4f}"
        )

    report = {
        "feature_version": settings.feature_version,
        "protocol": (
            "Method 7 configuration; alpha selected on validation using train-only SVM; "
            "SVM refit on train+validation; held-out test evaluated without hard filtering"
        ),
        "formula": "S_final = alpha*S_visual + (1-alpha)*S_SVM",
        "split": {
            "seed": settings.random_seed,
            "records": len(records),
            "species": len(species),
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "svm": {
            "hog_dimensions": hog_dimensions,
            "full_pca_target": full_pca_target,
            "C": c_value,
            "class_weight": "balanced",
            "validation_model_converged": validation_converged,
            "final_model_converged": final_converged,
            "method7_train_test_gap": selected["train_test_gap"],
            "fit_seconds": fit_seconds,
            "score_seconds_per_query": score_seconds_per_query,
        },
        "calibration_grid": calibration_grid,
        "selected_alpha_for_p1": p1_alpha,
        "selected_alpha_for_map10": map_alpha,
        "baseline_test": asdict(baseline_test),
        "svm_soft_test": svm_test,
        "weighted_knn_soft_same_test_reference": knn_reference,
    }
    report_path = settings.scaler_dir / "svm_soft_reranking_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
