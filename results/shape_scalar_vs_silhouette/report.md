# Shape Scalar vs BBox Silhouette (label-free)

## Protocol

- Gallery: 1468 images.
- Sources: 160; selection=80, held-out test=80.
- Queries: 1920 with 12 strong perturbations per source.
- No species labels were read.
- Selection score: 30% Hit@1 + 40% MRR + 30% RBO@10.

## Held-out comparison

| Configuration chosen on selection | Hit@1 | MRR | RBO@10 | Top-10 overlap | Composite |
|---|---:|---:|---:|---:|---:|
| HOG only | 0.9979 | 0.9985 | 0.6612 | 0.6921 | 0.8971 |
| Scalar only (shape-only) | 0.8250 | 0.8541 | 0.8099 | 0.8318 | 0.8321 |
| bbox_grid8 only (shape-only) | 0.9698 | 0.9807 | 0.8857 | 0.9033 | 0.9489 |
| HOG + Scalar s=0.50 | 0.9979 | 0.9986 | 0.6712 | 0.6984 | 0.9002 |
| HOG + bbox_grid8 b=1.00 | 0.9979 | 0.9990 | 0.7040 | 0.7355 | 0.9102 |
| HOG + Scalar + bbox_grid8 b=0.90 | 0.9979 | 0.9990 | 0.7035 | 0.7356 | 0.9100 |

## Decision

- Winner: **BBox Silhouette**.
- Shape Scalar marginal delta beside HOG: +0.0031.
- BBox Silhouette marginal delta beside HOG: +0.0130.
- Adding Shape Scalar after BBox Silhouette: -0.0001.
- Keep both: **NO** using the +0.0005 minimum material-gain rule.

BBox Silhouette beats the selected Scalar configuration on 10 of 12 perturbations. Scalar is better for a 5-degree rotation and slightly better for strong mask erosion. The winning `b=1.00` value is internal to the shape group and removes HOG from that experimental shape mix; it is not 100% of the complete retrieval score.

The result measures exact-source robustness and Top-10 stability, not species classification or human visual preference.
