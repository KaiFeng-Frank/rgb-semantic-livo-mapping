# Semantic-assisted pose refinement: seq07

Fixed geometry, initial trajectory, accepted loop, graph and predicted labels. One known sequence / one loop; offline, no new training. G/D/S/T differ only in registration weights. D: movable-class downweighting; S: plus semantic compatibility; T: plus temporal reliability. ZS/B0 are archived pass-1 predictions. Lower is better for all table errors.

| Arm | ATE m | RPE 1s m | RPE 1s deg | Loop m | Loop deg | Fallback |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0.836592 | 0.097061 | 0.299185 | 2.372079 | 2.993355 | False |
| open3d | 0.557214 | 0.096732 | 0.297796 | 0.319985 | 0.327763 | False |
| G | 0.557214 | 0.096732 | 0.297796 | 0.319985 | 0.327763 | False |
| ZS_D | 0.556136 | 0.096732 | 0.297801 | 0.319080 | 0.345067 | False |
| ZS_S | 0.555798 | 0.096731 | 0.297803 | 0.318137 | 0.349390 | False |
| ZS_T | 0.556265 | 0.096728 | 0.297800 | 0.321850 | 0.325634 | False |
| armB0_s1_D | 0.556808 | 0.096731 | 0.297798 | 0.320708 | 0.333688 | False |
| armB0_s1_S | 0.556708 | 0.096731 | 0.297798 | 0.320158 | 0.334548 | False |
| armB0_s1_T | 0.557178 | 0.096729 | 0.297794 | 0.324319 | 0.306418 | False |

![Pose comparison](pose-comparison.png)

![Position error](error-over-time.png)

All candidate trajectories, including any fallback, are scored separately in comparison.json. Do not interpret the best row as a validated deployment choice. Improvements in one error may trade against another.

registration-inputs.npz contains derived voxel centroids, labels/confidence and temporal evidence for reproduction, not raw scans or full prediction archives. measurements.json records every candidate, guard and graph solve. Protocol files and SHA256SUMS freeze the experiment. Ground truth is confined to score.py.
