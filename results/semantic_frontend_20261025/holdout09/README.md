# Frozen sequence 09 frontend validation

**Accuracy admission: failed. Boundary activation: passed.**

The first 240 frames were frozen before scoring, with five arms and three rotated repetitions. All 15 trajectories retain the same 235 timestamps. No parameters were changed after seeing these results. This is a frontend holdout; earlier segmentation work has used sequence 09.

| Arm | Mean ATE, m (repeat range) | Mean 1 s translation RPE, m | Mean 1 s rotation RPE, deg | Updates with boundary rows |
|---|---:|---:|---:|---:|
| geometry | 0.429595 (0.408818–0.449055) | 0.132349 | 0.166198 | 0/235 |
| association_B0 | 0.443236 (0.390982–0.542964) | 0.131500 | 0.165189 | 0/235 |
| boundary_B0 | 0.428703 (0.405052–0.461249) | 0.135766 | 0.167496 | 233/235 |
| full_B0 | 0.463356 (0.411198–0.506581) | 0.131606 | 0.169117 | 233/235 |
| full_ZS | 0.398232 (0.379615–0.425463) | 0.129208 | 0.159094 | 224/235 |

Full B0 mean ATE is 7.86% higher than geometry. Only one of three paired ATE differences improves; the repeat ranges overlap, and mean rotation RPE regresses. Boundary rows are active on 233/235 updates in every B0 repetition, so activation alone does not establish useful localization information.

Adding boundaries to association gives paired ATE changes -0.131766, +0.076528, +0.115599 m. Its mean ATE also increases, with overlapping ranges. The zero-shot full arm has lower mean ATE than geometry, but overlapping ranges; it does not satisfy the preregistered primary B0 comparison and is not selected as a replacement success.

This bounded cached-label result establishes neither whole-sequence accuracy nor live throughput. It does not admit a larger experiment. Loop closure was disabled, there was no training, and the runner used archived semantic predictions.

[All repetitions](scores.json), [frozen decision checks](assessment.json), [executed protocol](protocol.json), [full original logs and trajectories](github-actions-40156371477.tar.gz), [GitHub Actions run](https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/actions/runs/40156371477). The archive SHA256 is recorded in the assessment. Raw scans, images and model files are not included.
