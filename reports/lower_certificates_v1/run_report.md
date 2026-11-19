# Verified lower and upper class-risk assessment

Run: `lower-certificates-20261119-v1`.

The numerical witnesses and provenance were checked. Statistical conclusions assume independent identically distributed calibration episodes, independent of development.

The predictors and features are exact archived finite-support computational tables; historical model replay and global executable-readout regularity are not claimed.

| Class | n | Repetition | Lower | Class upper | Class-risk audit interval | Decision |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| raw_k2_1 | 1024 | 0 | 0.03005070 | 0.08820596 | [0.05625000, 0.05625000] | requirement_met |
| kahm_mem_k2_1 | 1024 | 0 | 0.09581997 | unavailable | [0.13353473, 0.13354830] | unresolved |
| kahm_local_k2_1 | 1024 | 0 | 0.03005070 | 0.08820596 | [0.05625000, 0.05625000] | requirement_met |
| short_history_k4_1 | 1024 | 0 | 0.11368306 | 0.20525653 | [0.15625000, 0.15625000] | class_excluded |
| full_history_k4_1 | 1024 | 0 | 0.03005070 | 0.08820596 | [0.05625000, 0.05625000] | requirement_met |
| raw_k2_1 | 4096 | 0 | 0.04281014 | 0.07227278 | [0.05625000, 0.05625000] | requirement_met |
| kahm_mem_k2_1 | 4096 | 0 | 0.11470296 | unavailable | [0.13353473, 0.13354830] | class_excluded |
| kahm_local_k2_1 | 4096 | 0 | 0.04281014 | 0.07227278 | [0.05625000, 0.05625000] | requirement_met |
| short_history_k4_1 | 4096 | 0 | 0.13476373 | 0.18078832 | [0.15625000, 0.15625000] | class_excluded |
| full_history_k4_1 | 4096 | 0 | 0.04281014 | 0.07227278 | [0.05625000, 0.05625000] | requirement_met |
| raw_k2_1 | 16384 | 0 | 0.04789719 | 0.06249135 | [0.05625000, 0.05625000] | requirement_met |
| kahm_mem_k2_1 | 16384 | 0 | 0.12376359 | unavailable | [0.13353473, 0.13354830] | class_excluded |
| kahm_local_k2_1 | 16384 | 0 | 0.04789719 | 0.06249135 | [0.05625000, 0.05625000] | requirement_met |
| short_history_k4_1 | 16384 | 0 | 0.14578659 | 0.16886171 | [0.15625000, 0.15625000] | class_excluded |
| full_history_k4_1 | 16384 | 0 | 0.04789719 | 0.06249135 | [0.05625000, 0.05625000] | requirement_met |

Decimal values are descriptive approximations. Exact fractions in the JSON records govern every decision.

Each block record reports both the normalized-block lower confidence bound and the empirical-optimum bound obtained from the same optimization witnesses. The full empirical bound is reported separately.

Use `python -m kahm_demo verify-class-risk --run PATH` to recompute witness checks without refitting or solving new optimization problems.
