# Label-free Handcrafted Feature Selection Report

## Protocol

- Feature version: `v4-color`.
- Gallery: 1468 original processed images.
- Sampled originals: 300; selection=150, test=150.
- Queries: 2100 (7 deterministic perturbations per original).
- Relevance target: the exact corresponding original image; species labels are never read or used.
- No classifier, PCA, deep embedding, species voting, database mutation, or web change.

## Baseline

| Split | Hit@1 | Hit@5 | MRR | Mean rank | Mean target distance |
|---|---:|---:|---:|---:|---:|
| Selection | 0.9990 | 1.0000 | 0.9995 | 1.00 | 0.0931 |
| Test | 0.9990 | 1.0000 | 0.9995 | 1.00 | 0.0929 |

The exact-original recovery metric is saturated on these mild perturbations. A neutral leave-one-out result therefore demonstrates redundancy for this robustness task, but does **not** by itself prove that a descriptor is useless for visual similarity between different images.

## Leave-One-Feature-Out and decision

| Feature | Status | ΔHit@1 | ΔMRR | Evidence |
|---|---|---:|---:|---|
| global_hsv | **NEEDS_MORE_TESTING** | +0.0000 | +0.0000 | Recovery is saturated; neutral removal shows redundancy, not visual uselessness. |
| spatial_hsv | **NEEDS_MORE_TESTING** | +0.0000 | +0.0000 | Recovery is saturated; neutral removal shows redundancy, not visual uselessness. |
| lbp | **REMOVE** | +0.0000 | +0.0000 | Weak alone, expensive to extract, and removal is neutral on both splits. |
| glcm | **REMOVE** | -0.0010 | -0.0005 | Weak alone and removal is neutral on selection while slightly better on test. |
| gabor | **NEEDS_MORE_TESTING** | +0.0038 | +0.0029 | Effect is small or inconsistent across selection and test. |
| silhouette | **NEEDS_MORE_TESTING** | +0.0000 | +0.0000 | Baseline weight is zero; use explicit silhouette-weight trials. |
| scalar | **NEEDS_MORE_TESTING** | +0.0000 | +0.0000 | Recovery is saturated; neutral removal shows redundancy, not visual uselessness. |
| hog | **NEEDS_MORE_TESTING** | +0.0000 | +0.0000 | Recovery is saturated; neutral removal shows redundancy, not visual uselessness. |

Silhouette weights 0.1, 0.2, and 0.3 all produced the same test MRR as Full (0.9995). This provides no evidence of a robustness gain, but the saturated task is insufficient to mark Silhouette as useless.

Current evidence supports testing a reduced candidate without **LBP and GLCM**. Global/Spatial HSV, Gabor, Silhouette, Shape Scalar, and HOG remain `NEEDS_MORE_TESTING` because exact-original recovery is too easy to establish their contribution to cross-image visual similarity. No production feature is removed by this experiment.

## Configuration results (held-out test)

| Configuration | Category | Hit@1 | Hit@5 | MRR | Mean rank | Dimensions | Total latency ms/query |
|---|---|---:|---:|---:|---:|---:|---:|
| Full | baseline | 0.9990 | 1.0000 | 0.9995 | 1.00 | 2411 | 162.134 |
| Full - global_hsv | leave_one_out | 0.9990 | 1.0000 | 0.9995 | 1.00 | 2123 | 152.572 |
| Full - spatial_hsv | leave_one_out | 0.9990 | 1.0000 | 0.9995 | 1.00 | 2195 | 151.371 |
| Full - lbp | leave_one_out | 0.9990 | 1.0000 | 0.9995 | 1.00 | 2383 | 64.467 |
| Full - glcm | leave_one_out | 1.0000 | 1.0000 | 1.0000 | 1.00 | 2401 | 152.681 |
| Full - gabor | leave_one_out | 0.9952 | 0.9981 | 0.9967 | 1.03 | 2375 | 139.994 |
| Full - silhouette | leave_one_out | 0.9990 | 1.0000 | 0.9995 | 1.00 | 2347 | 160.350 |
| Full - scalar | leave_one_out | 0.9990 | 1.0000 | 0.9995 | 1.00 | 2406 | 157.943 |
| Full - hog | leave_one_out | 0.9990 | 1.0000 | 0.9995 | 1.00 | 647 | 155.648 |
| global_hsv | single | 0.9981 | 0.9990 | 0.9986 | 1.02 | 288 | 9.571 |
| spatial_hsv | single | 0.9981 | 0.9990 | 0.9987 | 1.01 | 216 | 10.779 |
| lbp | single | 0.7000 | 0.7162 | 0.7090 | 153.18 | 28 | 97.686 |
| glcm | single | 0.5467 | 0.7248 | 0.6235 | 72.09 | 10 | 9.484 |
| gabor | single | 0.9933 | 0.9981 | 0.9957 | 1.17 | 36 | 22.184 |
| silhouette | single | 1.0000 | 1.0000 | 1.0000 | 1.00 | 64 | 1.821 |
| scalar | single | 0.8810 | 0.9819 | 0.9257 | 1.48 | 5 | 4.249 |
| hog | single | 0.9981 | 1.0000 | 0.9989 | 1.00 | 1764 | 6.516 |
| HOG + Gabor | complementarity | 1.0000 | 1.0000 | 1.0000 | 1.00 | 1800 | 28.676 |
| Spatial HSV + Global HSV | complementarity | 0.9981 | 0.9990 | 0.9987 | 1.01 | 504 | 20.325 |
| GLCM + LBP | complementarity | 0.5562 | 0.7286 | 0.6319 | 72.48 | 38 | 107.129 |
| HOG + Silhouette | complementarity | 1.0000 | 1.0000 | 1.0000 | 1.00 | 1828 | 8.314 |
| Full - LBP - Silhouette | reduced | 0.9990 | 1.0000 | 0.9995 | 1.00 | 2319 | 62.710 |
| Full - LBP - Silhouette - Global HSV | reduced | 0.9990 | 1.0000 | 0.9995 | 1.00 | 2031 | 53.132 |
| Full - LBP - Silhouette - GLCM | reduced | 1.0000 | 1.0000 | 1.0000 | 1.00 | 2309 | 53.234 |
| Full - LBP - Silhouette - Shape Scalar | reduced | 0.9990 | 1.0000 | 0.9995 | 1.00 | 2314 | 58.483 |
| Full silhouette=0.1 | silhouette_weight | 0.9990 | 1.0000 | 0.9995 | 1.00 | 2411 | 162.136 |
| Full silhouette=0.2 | silhouette_weight | 0.9990 | 1.0000 | 0.9995 | 1.00 | 2411 | 162.141 |
| Full silhouette=0.3 | silhouette_weight | 0.9990 | 1.0000 | 0.9995 | 1.00 | 2411 | 162.160 |

## Single Feature Test

| Feature | Hit@1 | Hit@5 | MRR | Mean rank | Dimensions |
|---|---:|---:|---:|---:|---:|
| Global HSV | 0.9981 | 0.9990 | 0.9986 | 1.02 | 288 |
| Spatial HSV | 0.9981 | 0.9990 | 0.9987 | 1.01 | 216 |
| LBP | 0.7000 | 0.7162 | 0.7090 | 153.18 | 28 |
| GLCM | 0.5467 | 0.7248 | 0.6235 | 72.09 | 10 |
| Gabor | 0.9933 | 0.9981 | 0.9957 | 1.17 | 36 |
| Silhouette | 1.0000 | 1.0000 | 1.0000 | 1.00 | 64 |
| Shape Scalar | 0.8810 | 0.9819 | 0.9257 | 1.48 | 5 |
| HOG | 0.9981 | 1.0000 | 0.9989 | 1.00 | 1764 |

These values measure stability and exact-original recovery only. In particular, perfect Silhouette recovery does not prove that silhouette alone produces visually desirable neighbors for different images.

## Complementarity Test

| Configuration | Hit@1 | Hit@5 | MRR | Dimensions |
|---|---:|---:|---:|---:|
| HOG + Gabor | 1.0000 | 1.0000 | 1.0000 | 1800 |
| Spatial HSV + Global HSV | 0.9981 | 0.9990 | 0.9987 | 504 |
| GLCM + LBP | 0.5562 | 0.7286 | 0.6319 | 38 |
| HOG + Silhouette | 1.0000 | 1.0000 | 1.0000 | 1828 |

HOG+Gabor and HOG+Silhouette both recover the original perfectly, but the saturated target prevents ranking one pair as visually superior without manual relevance judgments.

## Extraction and distance latency

| Feature | Dimensions | Extraction ms | Gallery-distance ms/query |
|---|---:|---:|---:|
| global_hsv | 288 | 5.314 | 4.237 |
| spatial_hsv | 216 | 7.529 | 3.223 |
| lbp | 28 | 97.228 | 0.429 |
| glcm | 10 | 9.429 | 0.017 |
| gabor | 36 | 22.115 | 0.037 |
| silhouette | 64 | 1.674 | 0.111 |
| scalar | 5 | 4.216 | 0.012 |
| hog | 1764 | 4.287 | 2.206 |

## Distance redundancy

Spearman correlation uses 20,000 deterministic random gallery pairs. High correlation is supporting evidence only, not a removal rule.

| Feature A | Feature B | Spearman |
|---|---|---:|
| global_hsv | spatial_hsv | 0.9307 |
| silhouette | hog | 0.7631 |
| scalar | hog | 0.4232 |
| silhouette | scalar | 0.3781 |
| lbp | glcm | 0.3549 |
| glcm | gabor | 0.2097 |
| gabor | hog | 0.1331 |
| gabor | scalar | 0.0839 |
| global_hsv | lbp | 0.0756 |
| spatial_hsv | lbp | 0.0640 |
| gabor | silhouette | 0.0620 |
| spatial_hsv | hog | 0.0606 |
| global_hsv | hog | 0.0527 |
| glcm | scalar | 0.0466 |
| global_hsv | glcm | 0.0420 |
| spatial_hsv | silhouette | 0.0397 |
| lbp | hog | 0.0389 |
| lbp | scalar | 0.0388 |
| spatial_hsv | gabor | -0.0359 |
| spatial_hsv | scalar | 0.0341 |
| global_hsv | gabor | -0.0335 |
| global_hsv | silhouette | 0.0293 |
| global_hsv | scalar | 0.0280 |
| glcm | hog | 0.0259 |
| lbp | silhouette | 0.0239 |
| spatial_hsv | glcm | 0.0163 |
| lbp | gabor | 0.0065 |
| glcm | silhouette | 0.0062 |

## Robustness by perturbation

| Transformation | Hit@1 | Hit@5 | MRR | Mean rank | Mean original distance |
|---|---:|---:|---:|---:|---:|
| Brightness +10% | 1.0000 | 1.0000 | 1.0000 | 1.00 | 0.0402 |
| Brightness −10% | 1.0000 | 1.0000 | 1.0000 | 1.00 | 0.0440 |
| Gaussian blur | 1.0000 | 1.0000 | 1.0000 | 1.00 | 0.1051 |
| Translate 5px | 1.0000 | 1.0000 | 1.0000 | 1.00 | 0.1050 |
| Scale +5% | 1.0000 | 1.0000 | 1.0000 | 1.00 | 0.1249 |
| Mask erosion | 0.9933 | 1.0000 | 0.9967 | 1.01 | 0.1152 |
| Mask dilation | 1.0000 | 1.0000 | 1.0000 | 1.00 | 0.1157 |

Mask erosion is the only tested perturbation that causes any Top-1 failures. `per_transform_metrics.csv` contains the same breakdown for every configuration.

## Visual Top-10 inspection

Open `visual_comparison.html`. It shows transformed queries and Top-10 results side by side. The report does not assign subjective visual relevance labels.

## Artifacts

- `configuration_summary.csv`
- `per_transform_metrics.csv`
- `per_query_results.csv`
- `feature_latency.csv`
- `distance_correlations.csv`
- `robustness_mrr.svg`
- `feature_extraction_latency.svg`
- `visual_comparison.html`

## Interpretation limits

Exact-original recovery measures perturbation stability, not semantic or species correctness. Manual visual inspection remains necessary for whether different images look meaningfully similar.
