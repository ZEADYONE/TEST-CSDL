# LBP and Silhouette Parameter Study (label-free)

## Protocol

- Gallery: 1468 images.
- Source images: 160; selection=80, held-out test=80.
- Queries: 1920 (12 stronger deterministic perturbations/source).
- No species names or labels were read.
- Selection score = 30% Hit@1 + 40% MRR + 30% RBO@10.
- The baseline contains HSV, GLCM, Gabor, shape scalars, and HOG, but no LBP or silhouette.

## Held-out result

| Configuration chosen on selection | Hit@1 | MRR | RBO@10 | Top-10 overlap | Composite |
|---|---:|---:|---:|---:|---:|
| No LBP / No Silhouette | 0.9979 | 0.9985 | 0.6635 | 0.6930 | 0.8978 |
| LBP default_r1 w=0.30 | 0.9979 | 0.9985 | 0.6633 | 0.6933 | 0.8978 |
| Silhouette bbox_grid8 w=0.80 | 0.9990 | 0.9995 | 0.6983 | 0.7278 | 0.9090 |
| Best validation LBP + Silhouette | 0.9990 | 0.9995 | 0.6978 | 0.7276 | 0.9088 |

## Decision

- LBP: **NO_MEASURABLE_GAIN**, held-out composite delta -0.0000.
- Silhouette: **SUPPORTED**, held-out composite delta +0.0112.
- Combined: **SUPPORTED**, held-out composite delta +0.0110.

Across all 36 non-zero LBP configurations, the composite delta ranged from -0.001196 to +0.000033 on selection and from -0.000389 to +0.000051 on held-out test. This is measurement-level variation, not evidence of useful contribution.

Across all 54 non-zero Silhouette configurations, the delta ranged from -0.002272 to +0.008963 on selection and from +0.000305 to +0.011159 on held-out test. BBox-normalized 4x4, 8x8, and 16x16 variants all showed the same positive trend.

The best silhouette is bbox-normalized, showing that the old full-canvas grid mixed true shape with absolute position and scale. `bbox_grid8 w=0.80` improved most perturbations, especially translation and scale, but reduced RBO@10 for rotation by 0.0486 and for strong mask erosion by 0.0048. Its weight is internal to the shape group: with a 0.40 shape-group weight, its nominal total contribution is 0.32, not 0.80 of the complete score.

These metrics test robustness and neighbor-list stability, not species classification and not human visual preference. A small positive delta should not justify production complexity by itself. See `configuration_summary.csv` for every parameter combination and `per_transform_test.csv` for failure modes.
