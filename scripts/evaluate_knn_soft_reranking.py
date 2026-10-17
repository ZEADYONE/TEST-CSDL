"""Evaluate weighted-kNN species evidence as a soft visual reranker.

The weighted-kNN configuration comes from Method 5. Only alpha is tuned here,
using calibration queries; the held-out test split is evaluated for the chosen
alpha only.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import joinedload

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models import ImageFeature, ImageRecord  # noqa: E402
from scripts.evaluate_species_prototypes import (  # noqa: E402
    group_raw_distances,
    normalize_and_fuse,
)
from scripts.evaluate_weighted_knn import weighted_species_scores  # noqa: E402
from scripts.tune_weights import (  # noqa: E402
    evaluate,
    metric_key,
    robust_normalize,
    split_queries,
)


ALPHAS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
EPSILON = 1e-10


def normalize_species_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize each query's species evidence to [0, 1]."""

    maximum = np.max(scores, axis=1, keepdims=True)
    return (scores / np.maximum(maximum, EPSILON)).astype(np.float32)


def soft_reranked_distance(
    visual_distance: np.ndarray,
    normalized_species_scores: np.ndarray,
    label_indices: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fuse scores without filtering candidates by their predicted species."""

    visual_similarity = np.clip(1.0 - visual_distance, 0.0, 1.0)
    candidate_species_similarity = normalized_species_scores[:, label_indices]
    final_similarity = (
        alpha * visual_similarity
        + (1.0 - alpha) * candidate_species_similarity
    )
    distance = (1.0 - final_similarity).astype(np.float32)
    np.fill_diagonal(distance, np.inf)
    return distance


def load_method5_config(report_path: Path) -> tuple[int, int, float]:
    if not report_path.is_file():
        raise FileNotFoundError(
            "Method 5 report is missing. Run scripts/evaluate_weighted_knn.py first."
        )
    selected = json.loads(report_path.read_text(encoding="utf-8"))["selected"]
    return int(selected["neighbors"]), int(selected["power"]), float(selected["beta"])


def main() -> int:
    settings = get_settings()
    method5_report = settings.scaler_dir / "weighted_knn_report.json"
    neighbors, power, beta = load_method5_config(method5_report)

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
    features = (
        np.asarray([record.feature.color_vector for record in records], dtype=np.float32),
        np.asarray([record.feature.texture_vector for record in records], dtype=np.float32),
        np.asarray([record.feature.shape_vector for record in records], dtype=np.float32),
    )
    calibration, test = split_queries(labels, settings.random_seed)
    print(
        f"Feature version={settings.feature_version}, records={len(records)}, "
        f"species={len(species)}, calibration={len(calibration)}, test={len(test)}"
    )
    print(f"Method 5 configuration: N={neighbors}, p={power}, beta={beta:g}")

    print("Computing current visual distances and weighted-kNN evidence...")
    visual_groups = group_raw_distances(features, features, settings)
    visual_ranges = [robust_normalize(group)[1]["p95"] for group in visual_groups]
    visual_distance = normalize_and_fuse(visual_groups, visual_ranges, settings)
    np.fill_diagonal(visual_distance, np.inf)
    raw_species_scores = weighted_species_scores(
        visual_distance,
        label_indices,
        species_counts,
        neighbors=neighbors,
        power=power,
        beta=beta,
    )
    species_scores = normalize_species_scores(raw_species_scores)

    calibration_grid: dict[str, dict[str, float]] = {}
    best_map_alpha: float | None = None
    best_map_calibration = None
    best_p1_alpha: float | None = None
    best_p1_calibration = None
    print(
        f"\n{'alpha':>7} {'visual':>8} {'species':>8} {'P@1':>8} "
        f"{'P@5':>8} {'P@10':>8} {'R@10':>8} {'mAP@10':>8}"
    )
    for alpha in ALPHAS:
        distance = soft_reranked_distance(
            visual_distance, species_scores, label_indices, alpha
        )
        metrics = evaluate(distance, labels, calibration)
        calibration_grid[f"{alpha:.1f}"] = asdict(metrics)
        print(
            f"{alpha:7.1f} {alpha:8.1f} {1.0-alpha:8.1f} "
            f"{metrics.p1:8.4f} {metrics.p5:8.4f} {metrics.p10:8.4f} "
            f"{metrics.recall10:8.4f} {metrics.map10:8.4f}"
        )
        if best_map_calibration is None or metric_key(metrics) > metric_key(
            best_map_calibration
        ):
            best_map_alpha = alpha
            best_map_calibration = metrics
        p1_key = (metrics.p1, metrics.map10, metrics.p5)
        if best_p1_calibration is None or p1_key > (
            best_p1_calibration.p1,
            best_p1_calibration.map10,
            best_p1_calibration.p5,
        ):
            best_p1_alpha = alpha
            best_p1_calibration = metrics

    assert best_map_alpha is not None and best_map_calibration is not None
    assert best_p1_alpha is not None and best_p1_calibration is not None
    baseline_test = evaluate(visual_distance, labels, test)
    map_test = evaluate(
        soft_reranked_distance(
            visual_distance, species_scores, label_indices, best_map_alpha
        ),
        labels,
        test,
    )
    p1_test = evaluate(
        soft_reranked_distance(
            visual_distance, species_scores, label_indices, best_p1_alpha
        ),
        labels,
        test,
    )

    print(
        f"\nSelected on calibration: alpha={best_map_alpha:.1f} for mAP@10; "
        f"alpha={best_p1_alpha:.1f} for P@1"
    )
    print(
        f"Test baseline: P@1={baseline_test.p1:.4f}, P@5={baseline_test.p5:.4f}, "
        f"P@10={baseline_test.p10:.4f}, R@10={baseline_test.recall10:.4f}, "
        f"mAP@10={baseline_test.map10:.4f}"
    )
    print(
        f"Test alpha={best_p1_alpha:.1f} (P@1 policy): P@1={p1_test.p1:.4f}, "
        f"P@5={p1_test.p5:.4f}, P@10={p1_test.p10:.4f}, "
        f"R@10={p1_test.recall10:.4f}, mAP@10={p1_test.map10:.4f}"
    )
    print(
        f"Test alpha={best_map_alpha:.1f} (mAP policy): P@1={map_test.p1:.4f}, "
        f"P@5={map_test.p5:.4f}, P@10={map_test.p10:.4f}, "
        f"R@10={map_test.recall10:.4f}, mAP@10={map_test.map10:.4f}"
    )

    report = {
        "feature_version": settings.feature_version,
        "protocol": (
            "leave-one-out visual retrieval; Method 5 weighted-kNN configuration; "
            "per-query max-normalized species evidence; tune alpha on calibration only"
        ),
        "formula": "S_final = alpha*S_visual + (1-alpha)*S_species",
        "split": {
            "seed": settings.random_seed,
            "records": len(records),
            "species": len(species),
            "calibration": len(calibration),
            "test": len(test),
        },
        "weighted_knn": {
            "neighbors": neighbors,
            "power": power,
            "beta": beta,
            "source": str(method5_report),
        },
        "selection_metrics": {
            "map_policy": "calibration mAP@10, then P@1/P@5",
            "p1_policy": "calibration P@1, then mAP@10/P@5",
        },
        "calibration_grid": calibration_grid,
        "baseline_test": asdict(baseline_test),
        "map_policy": {
            "selected_alpha": best_map_alpha,
            "calibration": asdict(best_map_calibration),
            "test": asdict(map_test),
        },
        "p1_policy": {
            "selected_alpha": best_p1_alpha,
            "calibration": asdict(best_p1_calibration),
            "test": asdict(p1_test),
        },
    }
    report_path = settings.scaler_dir / "knn_soft_reranking_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
