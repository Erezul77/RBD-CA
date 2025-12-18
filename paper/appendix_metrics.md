# Appendix: Metrics and Excerpt

## Metric definitions (from code)
- **identity_score**: persist_iou when available; otherwise core_jaccard_prev; else -1.
- **persist_iou**: hit rate of current core against a dilated halo of the previous core (persistence bias).
- **core_jaccard_prev**: Jaccard overlap of current vs previous core.
- **core_centroid_drift**: Euclidean distance between attention center and core centroid.
- **boundary_to_active**: boundary cell count divided by active cell count (roughness).
- **core_size_delta**: core_size minus baseline core_size measured at perturb_t (post-shock recovery); NaN before/without shock.
- **observer_action_rate**: applied overrides inside window divided by window size (active mode only).
- **identity_break**: 1 when core_jaccard_prev drops below identity_break_thresh; **identity_run_len** counts steps since last break.
- **boundary_mode**, **observer_mode**, **tag**: run metadata recorded each step.

## Excerpt: `analysis_outputs/sweep_summary_by_mode.csv` (first 8 rows)
```
boundary_mode,observer_mode,mean_identity_score_mean,mean_identity_score_std,final_identity_score_mean,final_identity_score_std,mean_recovery_mean,mean_recovery_std,mean_action_rate_mean,mean_action_rate_std
zero,passive_nopersist,0.0003001153183657199,0.0003953889445780224,0.0,0.0,6.267272727272727,6.504271450934225,0.0,0.0
zero,object_off_tracked,0.6472260830166953,0.006180906754750832,0.4305072417922169,0.3317270925256871,384.954,1173.4391262541062,0.0,0.0
zero,passive_persist,0.6921881009572458,0.008514790315388578,0.7333006224047726,0.31691607363317553,35.883,151.2420436287476,0.0,0.0
zero,active_persist,0.703808008027485,0.011261856187864823,0.7386496906963049,0.28174232522814413,-70.05699999999999,120.46841885324136,0.049671292914536154,0.0
wrap,object_off_tracked,0.6108009574343017,0.009806580530654667,0.5278456605165881,0.28026432026620907,-596.2439999999999,1418.8922421466684,0.0,0.0
wrap,passive_nopersist,0.00029027680318580564,0.0003823532831888967,0.0,0.0,-17.946,83.65208116956805,0.0,0.0
wrap,passive_persist,0.6792446677992936,0.012537293114458404,0.5249889628562457,0.23360288894068246,16.279,206.96554686468954,0.0,0.0
wrap,active_persist,0.6906425179050328,0.011589710737539284,0.8112461098826923,0.16838732602133738,112.68700000000001,138.53952548280222,0.049671292914536154,0.0
```

