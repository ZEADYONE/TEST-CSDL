# Label-free Hu Moments and Spatial Multi-scale LBP Experiment

## Protocol

- Gallery: 1468 processed originals.
- Sampled originals: 120; deterministic queries: 840.
- Selection/test split is random by image identity and never uses species.
- Exact-image recovery and Top-10 neighbor stability are separate metrics.
- The corresponding original is excluded before Top-10 stability is measured.
- No classifier, species voting, PCA, deep feature, IVF, database write, or web change.

## Held-out test results

| Configuration | Dim | Hit@1 | MRR | Top-10 overlap | Jaccard@10 | RBO@10 | Extract ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current Full | 2411 | 1.0000 | 1.0000 | 0.8317 | 0.7321 | 0.8176 | 130.28 |
| Current Full - LBP | 2383 | 1.0000 | 1.0000 | 0.8290 | 0.7283 | 0.8170 | 46.28 |
| Current Full + Hu 0.05 | 2418 | 1.0000 | 1.0000 | 0.8310 | 0.7303 | 0.8145 | 130.93 |
| Current Full + Hu 0.10 | 2418 | 1.0000 | 1.0000 | 0.8290 | 0.7285 | 0.8085 | 130.93 |
| Current Full + Hu 0.20 | 2418 | 1.0000 | 1.0000 | 0.8331 | 0.7333 | 0.8135 | 130.93 |
| Replace LBP with Spatial MS-LBP | 2545 | 1.0000 | 1.0000 | 0.8317 | 0.7321 | 0.8174 | 206.41 |
| Spatial MS-LBP + Hu 0.10 | 2552 | 1.0000 | 1.0000 | 0.8293 | 0.7289 | 0.8088 | 207.06 |

Top-10 overlap measures whether a perturbed query preserves the same visual neighbors as its original. It does not use or imply species correctness.

## Experimental feature cost

| Feature | Dim | Gallery extraction ms | Query extraction ms | Distance ms/query |
|---|---:|---:|---:|---:|
| spatial_ms_lbp | 162 | 161.34 | 160.13 | 3.091 |
| hu | 7 | 0.61 | 0.65 | 0.013 |

## Automatic conclusion boundary

- Baseline test RBO@10: 0.8176.
- Highest test RBO@10: Current Full (0.8176).
- A descriptor is not declared visually better from stability alone. Open `visual_comparison.html` and complete `manual_visual_ratings.csv`.
- Only keep an experimental descriptor if it improves held-out stability and human visual ratings enough to justify its latency.

## Experimental decisions

| Proposal | Status | Test evidence | Cost evidence |
|---|---|---|---|
| Remove current LBP | **NEEDS_MORE_TESTING** | RBO@10 0.8170 vs baseline 0.8176; nearly neutral | Saves 83.99 ms/query |
| Add Hu 0.05 | **NEEDS_MORE_TESTING** | RBO@10 0.8145, lower than baseline | Adds 0.65 ms/query |
| Add Hu 0.10 | **DO_NOT_ADOPT** | RBO@10 0.8085, lower than baseline | Adds 0.65 ms/query |
| Add Hu 0.20 | **NEEDS_MORE_TESTING** | Overlap 0.8331 is slightly higher, but RBO@10 0.8135 is lower | Adds 0.65 ms/query |
| Replace with Spatial MS-LBP | **DO_NOT_ADOPT** | RBO@10 0.8174, effectively unchanged/slightly lower | Adds 76.13 ms/query and 134 dimensions |
| Spatial MS-LBP + Hu 0.10 | **DO_NOT_ADOPT** | RBO@10 0.8088, clearly lower | Adds 76.78 ms/query and 141 dimensions |

These statuses concern the tested implementations and weights, not the entire Hu or LBP feature families. Human visual ratings are still required before a final production removal decision.

## Artifacts

- `configuration_summary.csv`
- `per_transform_metrics.csv`
- `experimental_feature_latency.csv`
- `visual_comparison.html`
- `manual_visual_ratings.csv`
