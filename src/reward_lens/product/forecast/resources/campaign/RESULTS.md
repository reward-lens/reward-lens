# Campaign results: the scored ledger

Predictions frozen 2026-07-18T23:46:57.951556+00:00 at git `f93f4b5578c221d3d2665f9b305f3e86759be862+dirty`; every threshold below predates the confirmatory evidence by construction (spec hashes verified against the registry rebuild at adjudication time).

19 of 27 frozen cards adjudicated against the merged evidence store; 16 of 53 frozen hypotheses confirmed, 21 refuted, 16 inconclusive; 8 kill criteria fired, counting the meta-ledger throughout. These counts are derived from the adjudication records cited below; the per-card table repeats them one row at a time.
Refutations below are reported with the same prominence as confirmations; each fired kill criterion is a registered negative result, not a failure of the machinery.

## Low-power acceptances recorded at the freeze

The freeze manifest records the cards whose sub-0.8 power at the frozen threshold was explicitly accepted rather than resolved; the power and MDE values below are transcribed from specs/frozen/manifest.json. Each card's section repeats this disclosure, and any confirmed headline sitting between its threshold and the recorded MDE is flagged there as a low-power-band detection.

- ATLAS-VCE: frozen at power 0.48 with MDE 0.00087371 (the vce > 0 margin sits at the null center by construction: the point call passes under the null 0.48 of the time and its power at the threshold-alternative equals that rate; the alpha-exact p-value companion and the registered CI exclusion carry the card's evidence).
- CHI-DRIFT: frozen at power 0.13 with MDE 0.908994 (split-bank estimator: chi and drift read disjoint prompt halves, so the null re-centers at zero, but the correlation's exchangeable unit is k = 6 features and the null still passes the frozen 0.3 threshold 0.28 of the time; the registered CI exclusion carries size control and only orderings near the synthetic calibration (0.958) clear the 0.91 MDE).
- EMB-LORA: frozen at power 0.50 with MDE 2.60464 (entry-ratio point call at the frozen > 1.0, the null median: power equals the null pass rate 0.501, ratios at or above 2.60 confirm at 80 percent, and the s07 plant sits far above that; passes only through the explicit accept-underpowered path).
- ERR-RB2: frozen at power 0.23 with MDE 0.0681264 (delta-AUROC point call at its own 0.03 threshold under the label-shuffle refit null (null sd 0.0235, null pass rate at threshold 0.2013); the registered CI exclusion carries size control and deltas at or above the MDE confirm at 80 percent).
- FACT-KUI: frozen at power 0.60 with MDE 0.294628 (fleet-median star KUI at n = 10 models over a 6-property battery; the property-random null passes the frozen 0.2 0.03 of the time, the K-kui permutation kill carries size control, and medians at or above 0.29 confirm at 80 percent).
- FORECAST-CAT: frozen at power 0.28 with MDE 0.0380599 (P(observed MAE beats the no-skill null's 5 percent boundary 0.048 | true MAE at the frozen 0.06), 6 categories; a no-skill forecast passes the frozen bound itself 0.100 of the time, and true MAEs at or below 0.038 detect at 80 percent).
- FORENSIC-RECEIPT: frozen at power 0.51 with MDE 0.674 (fraction of positive grader contrasts at n = 5 graders: the frozen 0.5 sits at the null center, the point call passes under the null 0.51 of the time, and 80 percent confirmation needs per-grader positive-contrast probability 0.67, the s15-plant regime (calibration 0.993)).
- PPE-BON: frozen at power 0.26 with MDE 0.641578 (Fisher-z detection of the frozen 0.3 against the null-critical 0.46  over n = 15 (model, set) points; the permutation null passes the frozen 0.3 0.17 of the time, the registered CI exclusion carries size control, and effects at or above 0.64 confirm at 80 percent).

## The campaign scored against itself (C0)

The Brier score over the campaign's [[claim value=16 ev=ev:3990df7af1488c08ad62786d879b0b0f field=metrics.n_directional tol=0.00032]] pre-assigned directional calls is [[claim value=0.26 ev=ev:3990df7af1488c08ad62786d879b0b0f field=metrics.brier_directional tol=5.2e-06]], which did not beat the always-guess-half coin's Brier of [[claim value=0.25 ev=ev:3990df7af1488c08ad62786d879b0b0f field=thresholds.H-brier tol=5e-06]]. Interval coverage over the [[claim value=4 ev=ev:3990df7af1488c08ad62786d879b0b0f field=metrics.n_intervals tol=8e-05]] genuine interval predictions is [[claim value=0.75 ev=ev:3990df7af1488c08ad62786d879b0b0f field=metrics.interval_coverage_0p8 tol=1.5e-05]] against a registered nominal rate of 0.8; at this count coverage is a companion diagnostic, not a kill metric.
The meta kill criterion fired (K-meta): the campaign's directional calls do not calibrate, and every confirmation above should be read with that in mind.

Interval hits, individually: CAL-TRANSFER:H-transfer: outside, ladder-8b-chi-teacher-variance: inside, ladder-8b-crystallization: inside, ladder-8b-snr: inside.

## Per-card verdicts

| Card | Verdict | Hypotheses (confirmed/refuted/inconclusive) | Kill fired | Adjudication record |
|---|---|---|---|---|
| ERR-RB2 | refuted | 0/2/0 | K-err | ev:6bf316c059230ff4ff99b118b1664631 |
| FORECAST-CAT | mixed | 1/1/0 | - | ev:1acdc14e2fa16124786a9cd0d44a03b2 |
| STYLE-RMB | inconclusive | 0/0/3 | - | ev:14bc4c666b5b115cf1fd2c86cf209f01 |
| PPE-BON | inconclusive | 0/0/2 | - | ev:5cd71da5713f2b2fb77697318b842401 |
| LADDER | mixed | 1/1/0 | - | ev:d746d48eef7a5630cecbd3e97a9d9cb8 |
| CHI-DRIFT | confirmed | 2/0/0 | - | ev:c978e3958769ecf93dcf079771a3aedb |
| HUMP | inconclusive | 0/0/2 | - | ev:02a1669641150750737afc8291255bf9 |
| GAUGE-E19 | inconclusive | 0/0/3 | - | ev:0b9b34f7e2af83d3b0ca32a60791a859 |
| GAUGE-XFAM | inconclusive | 0/0/2 | - | ev:095096260d9209c4fb9935cbb215812b |
| HACK-FORE | inconclusive | 0/0/1 | - | ev:137077efe204a6daddd359f54c494453 |
| SURGERY | mixed | 1/1/0 | - | ev:41058731abf1d0b1590d4f2286e8771c |
| VERIF-PRM | refuted | 1/1/0 | K-verif | ev:46e1500c4d39f9821c137ae0b00ac0da |
| JUDGE-VBC | confirmed | 1/0/0 | - | ev:e7fd3fee89a0c56cf4465212601d0d26 |
| VALUES-CONTEST | inconclusive | 0/0/2 | - | ev:e036326c47c7a0f52330559479e03790 |
| FORENSIC-RECEIPT | refuted | 0/2/0 | K-forensic | ev:426a8cbed7f6d6dfb61d8fe3ed5264a4 |
| ATLAS-VCE | refuted | 1/1/0 | K-vce | ev:0c80d9086ebb0ad9704b88a8eefb01fa |
| FACT-KUI | refuted | 0/3/0 | K-kui | ev:a34316feccee3beaed93c05ce5888567 |
| CAPACITY-WELCH | mixed | 1/2/0 | - | ev:f9e6a003a5a1e0aa0fe3ff1eca0d9c7b |
| EVAL-AWARE | mixed | 1/2/0 | - | ev:54fb272fd011c8ca07a4959f553341ab |
| TOPO-HODGE | confirmed | 1/0/0 | - | ev:844ad06a5aeb1d89c21c13c19a28cbd2 |
| EMB-LORA | mixed | 1/1/0 | - | ev:01c81c7185640f9b962b9115f360732e |
| CAL-TRANSFER | refuted | 0/1/0 | K-transfer | ev:b37f71c015ebfd697d7deb2a7efb5f7e |
| ADJ-AVP | refuted | 0/1/0 | K-avp | ev:a34457e83cd4eec10d080804a2533e82 |
| CONF-PARTIAL | mixed | 2/1/0 | - | ev:491f7f378d0b58dfdadebc9aaee7c983 |
| T3-DECOMP | confirmed | 1/0/0 | - | ev:a7ed2d6b81c5f79458dad733defb3803 |
| T3-FIELD | inconclusive | 0/0/1 | - | ev:9d51086b170b359d58d8eb4bf4f882fb |
| META-LEDGER | refuted | 1/1/0 | K-meta | ev:3990df7af1488c08ad62786d879b0b0f |

## Headline numbers, per card

### ERR-RB2: Error anticipation on RewardBench 2

This card was frozen under an explicit low-power acceptance: power 0.23 at MDE 0.0681264 (recorded in the freeze manifest); a confirmation between the frozen threshold and the MDE is a low-power detection, not a large-effect one.

- H-err-qwen (refuted): internal features beat the exposed seen-pair margin at predicting held-out RB2 completion errors on Qwen3-8B; `err_auroc_delta_qwen8b` = [[claim value=-0.000973843 ev=ev:6bf316c059230ff4ff99b118b1664631 field=metrics.err_auroc_delta_qwen8b tol=1e-06]] against the registered `>` [[claim value=0.03 ev=ev:6bf316c059230ff4ff99b118b1664631 field=thresholds.H-err-qwen tol=1e-06]].
- H-err-llama (refuted): the same holds on Llama-8B (cross-family); `err_auroc_delta_llama8b` = [[claim value=-0.00347074 ev=ev:6bf316c059230ff4ff99b118b1664631 field=metrics.err_auroc_delta_llama8b tol=1e-06]] against the registered `>` [[claim value=0.03 ev=ev:6bf316c059230ff4ff99b118b1664631 field=thresholds.H-err-llama tol=1e-06]].
- Kill K-err FIRED on `err_auroc_delta_min_family` = [[claim value=-0.00347074 ev=ev:6bf316c059230ff4ff99b118b1664631 field=metrics.err_auroc_delta_min_family tol=1e-06]] (`<=` [[claim value=0 ev=ev:6bf316c059230ff4ff99b118b1664631 field=kill_thresholds.K-err tol=1e-06]]): internals add nothing over the seen-pair margin the RM already exposes when predicting held-out errors, on at least one family
- Registered companions: `err_auroc_internal_qwen8b` = [[claim value=0.858056 ev=ev:6bf316c059230ff4ff99b118b1664631 field=metrics.err_auroc_internal_qwen8b tol=1.7e-05]]; `err_auroc_margin_qwen8b` = [[claim value=0.85903 ev=ev:6bf316c059230ff4ff99b118b1664631 field=metrics.err_auroc_margin_qwen8b tol=1.7e-05]]; `err_auroc_internal_llama8b` = [[claim value=0.87281 ev=ev:6bf316c059230ff4ff99b118b1664631 field=metrics.err_auroc_internal_llama8b tol=1.7e-05]]; `err_auroc_margin_llama8b` = [[claim value=0.87628 ev=ev:6bf316c059230ff4ff99b118b1664631 field=metrics.err_auroc_margin_llama8b tol=1.8e-05]].

### FORECAST-CAT: Benchmark forecasting from internals plus a 200-pair slice

This card was frozen under an explicit low-power acceptance: power 0.28 at MDE 0.0380599 (recorded in the freeze manifest); a confirmation between the frozen threshold and the MDE is a low-power detection, not a large-effect one.

- H-forecast-mae (refuted): internals plus a 200-pair slice forecast per-category RB2 accuracy within 0.06 MAE; `forecast_mae` = [[claim value=0.0869584 ev=ev:1acdc14e2fa16124786a9cd0d44a03b2 field=metrics.forecast_mae tol=1.7e-06]] against the registered `<` [[claim value=0.06 ev=ev:1acdc14e2fa16124786a9cd0d44a03b2 field=thresholds.H-forecast-mae tol=1.2e-06]].
- H-forecast-beats-size (confirmed): the forecast beats predict-from-size; `forecast_mae_minus_size_baseline` = [[claim value=-0.10242 ev=ev:1acdc14e2fa16124786a9cd0d44a03b2 field=metrics.forecast_mae_minus_size_baseline tol=2e-06]] against the registered `<` [[claim value=0 ev=ev:1acdc14e2fa16124786a9cd0d44a03b2 field=thresholds.H-forecast-beats-size tol=1e-06]].
- Registered companions: `size_baseline_mae` = [[claim value=0.189379 ev=ev:1acdc14e2fa16124786a9cd0d44a03b2 field=metrics.size_baseline_mae tol=3.8e-06]]; `slice_baseline_mae` = [[claim value=0.177015 ev=ev:1acdc14e2fa16124786a9cd0d44a03b2 field=metrics.slice_baseline_mae tol=3.5e-06]].

### STYLE-RMB: Bias battery predicts RM-Bench Hard ranking

Inconclusive: missing intermediate: no intermediate 'campaign.bias.battery' for roster_key='armorm' slice='diagnostic-v3-degradation'; the arc that produces it has not run or its shard was not merged.

### PPE-BON: Tail index forecasts the best-of-k plateau on PPE

This card was frozen under an explicit low-power acceptance: power 0.26 at MDE 0.641578 (recorded in the freeze manifest); a confirmation between the frozen threshold and the MDE is a low-power detection, not a large-effect one.

Inconclusive: missing intermediate: no intermediate 'campaign.scores' for roster_key='skywork-v2-qwen3-8b' slice='ppe-best-of-k::gpqa'; the arc that produces it has not run or its shard was not merged.

### LADDER: Same-recipe scale ladder with frozen 8B extrapolation

- H-ladder-coverage (confirmed): at least 2 of 3 capture-only internal quantities land in their pre-frozen 8B intervals; `ladder_8b_metrics_in_interval` = [[claim value=3 ev=ev:d746d48eef7a5630cecbd3e97a9d9cb8 field=metrics.ladder_8b_metrics_in_interval tol=6e-05]] against the registered `>=` [[claim value=2 ev=ev:d746d48eef7a5630cecbd3e97a9d9cb8 field=thresholds.H-ladder-coverage tol=4e-05]].
- H-ladder-beats-flat (refuted): the extrapolation beats a no-trend (equal-to-4B) baseline; `ladder_extrap_minus_flat_error` = [[claim value=0.600228 ev=ev:d746d48eef7a5630cecbd3e97a9d9cb8 field=metrics.ladder_extrap_minus_flat_error tol=1.2e-05]] against the registered `<` [[claim value=0 ev=ev:d746d48eef7a5630cecbd3e97a9d9cb8 field=thresholds.H-ladder-beats-flat tol=1e-06]].
- Registered companions: `ladder_8b_crystallization_in_interval` = [[claim value=1 ev=ev:d746d48eef7a5630cecbd3e97a9d9cb8 field=metrics.ladder_8b_crystallization_in_interval tol=2e-05]]; `ladder_8b_chi_teacher_in_interval` = [[claim value=1 ev=ev:d746d48eef7a5630cecbd3e97a9d9cb8 field=metrics.ladder_8b_chi_teacher_in_interval tol=2e-05]]; `ladder_8b_snr_in_interval` = [[claim value=1 ev=ev:d746d48eef7a5630cecbd3e97a9d9cb8 field=metrics.ladder_8b_snr_in_interval tol=2e-05]].

### CHI-DRIFT: Susceptibility chi predicts best-of-n feature drift on a real bank

This card was frozen under an explicit low-power acceptance: power 0.13 at MDE 0.908994 (recorded in the freeze manifest); a confirmation between the frozen threshold and the MDE is a low-power detection, not a large-effect one.

- H-chi-bon (confirmed): chi predicts realized best-of-n feature-drift ordering on a real bank; `chi_bon_spearman` = [[claim value=0.964286 ev=ev:c978e3958769ecf93dcf079771a3aedb field=metrics.chi_bon_spearman tol=1.9e-05]] against the registered `>` [[claim value=0.3 ev=ev:c978e3958769ecf93dcf079771a3aedb field=thresholds.H-chi-bon tol=6e-06]].
- H-chi-beats-var (confirmed): chi beats a base-variance-only baseline; `chi_minus_variance_baseline_spearman` = [[claim value=0.571429 ev=ev:c978e3958769ecf93dcf079771a3aedb field=metrics.chi_minus_variance_baseline_spearman tol=1.1e-05]] against the registered `>` [[claim value=0 ev=ev:c978e3958769ecf93dcf079771a3aedb field=thresholds.H-chi-beats-var tol=1e-06]].
- Registered companions: `chi_bon_spearman_second_model` = [[claim value=1 ev=ev:c978e3958769ecf93dcf079771a3aedb field=metrics.chi_bon_spearman_second_model tol=2e-05]].

### HUMP: The Goodhart hump located from a base-policy tail index

KL in this card is the Beirami best-of-n bound, reported as a bound on the divergence, not the classical log-n identity; every KL number in this section inherits that reading.

Inconclusive: missing intermediate: no intermediate 'campaign.recorder.drift' for roster_key='policy-qwen25-1.5b' slice='hump-prompts'; the arc that produces it has not run or its shard was not merged.

### GAUGE-E19: The two-fine-tunes gauge result, settled

Inconclusive: resolve_subjects failed: PermissionError(13, 'Permission denied').

### GAUGE-XFAM: Canonical cross-family angles predict pairwise disagreement

Inconclusive: resolve_subjects failed: PermissionError(13, 'Permission denied').

### HACK-FORE: Weights-derived indices flag the exploitable hack family before any attack

Inconclusive: missing intermediate: no intermediate 'campaign.index.table' for roster_key='armorm' slice='diagnostic-v3'; the arc that produces it has not run or its shard was not merged.

### SURGERY: Certified erasure removes the exploit without benchmark loss

- H-erase (confirmed): LEACE erasure of the flagged direction removes the exploit; `exploit_drift_reduction` = [[claim value=0.885616 ev=ev:41058731abf1d0b1590d4f2286e8771c field=metrics.exploit_drift_reduction tol=1.8e-05]] against the registered `>` [[claim value=0.8 ev=ev:41058731abf1d0b1590d4f2286e8771c field=thresholds.H-erase tol=1.6e-05]].
- H-retain (refuted): benchmark accuracy is retained through the erasure; `rb2_accuracy_delta_after_erasure` = [[claim value=-0.398752 ev=ev:41058731abf1d0b1590d4f2286e8771c field=metrics.rb2_accuracy_delta_after_erasure tol=8e-06]] against the registered `>` [[claim value=-0.01 ev=ev:41058731abf1d0b1590d4f2286e8771c field=thresholds.H-retain tol=1e-06]].
- Registered companions: `erasure_certificate_pass` = [[claim value=1 ev=ev:41058731abf1d0b1590d4f2286e8771c field=metrics.erasure_certificate_pass tol=2e-05]]; `probe_recovery_auc` = [[claim value=0.247619 ev=ev:41058731abf1d0b1590d4f2286e8771c field=metrics.probe_recovery_auc tol=5e-06]].

### VERIF-PRM: Does the process reward model verify or style-read

- H-verif-loc (refuted): the PRM localizes ProcessBench error steps; `dense_localization_auc` = [[claim value=0.282144 ev=ev:46e1500c4d39f9821c137ae0b00ac0da field=metrics.dense_localization_auc tol=5.6e-06]] against the registered `>` [[claim value=0.7 ev=ev:46e1500c4d39f9821c137ae0b00ac0da field=thresholds.H-verif-loc tol=1.4e-05]].
- H-verif-style (confirmed): the correctness preference is anchored, not style-carried; `style_share` = [[claim value=-0.426847 ev=ev:46e1500c4d39f9821c137ae0b00ac0da field=metrics.style_share tol=8.5e-06]] against the registered `<` [[claim value=0.5 ev=ev:46e1500c4d39f9821c137ae0b00ac0da field=thresholds.H-verif-style tol=1e-05]].
- Kill K-verif FIRED on `dense_localization_auc` = [[claim value=0.282144 ev=ev:46e1500c4d39f9821c137ae0b00ac0da field=metrics.dense_localization_auc tol=5.6e-06]] (`<` [[claim value=0.55 ev=ev:46e1500c4d39f9821c137ae0b00ac0da field=kill_thresholds.K-verif tol=1.1e-05]]): methodological kill: the instrument failed on a real PRM, a works-ledger finding, not a scientific negative
- Registered companions: `verification_score` = [[claim value=0.833504 ev=ev:46e1500c4d39f9821c137ae0b00ac0da field=metrics.verification_score tol=1.7e-05]].

### JUDGE-VBC: Verdict before critique on a real generative judge

- H-vbc (confirmed): the verdict decoded at the pre-critique position matches the final verdict; `verdict_prefix_match_rate` = [[claim value=1 ev=ev:e7fd3fee89a0c56cf4465212601d0d26 field=metrics.verdict_prefix_match_rate tol=2e-05]] against the registered `>=` [[claim value=0.9 ev=ev:e7fd3fee89a0c56cf4465212601d0d26 field=thresholds.H-vbc tol=1.8e-05]].
- Registered companions: `flip_rate` = [[claim value=0.822 ev=ev:e7fd3fee89a0c56cf4465212601d0d26 field=metrics.flip_rate tol=1.6e-05]].

### VALUES-CONTEST: Contested-direction loading predicts human rater disagreement

Inconclusive: missing intermediate: no intermediate 'campaign.scores' for roster_key='armorm' slice='helpsteer3-pref-2000'; the arc that produces it has not run or its shard was not merged.

### FORENSIC-RECEIPT: Skepticism and receipt reliance predict who rewards fabricated receipts

This card was frozen under an explicit low-power acceptance: power 0.51 at MDE 0.674 (recorded in the freeze manifest); a confirmation between the frozen threshold and the MDE is a low-power detection, not a large-effect one.

- H-reliance (refuted): the two group contrasts separate graders as their planted values would; `reliance_recovery` = [[claim value=0 ev=ev:426a8cbed7f6d6dfb61d8fe3ed5264a4 field=metrics.reliance_recovery tol=1e-06]] against the registered `>` [[claim value=0.5 ev=ev:426a8cbed7f6d6dfb61d8fe3ed5264a4 field=thresholds.H-reliance tol=1e-05]].
- H-credulity (refuted): the omission-credulity gap is real on constructed triples; `omission_credulity_gap` = [[claim value=0.0232563 ev=ev:426a8cbed7f6d6dfb61d8fe3ed5264a4 field=metrics.omission_credulity_gap tol=1e-06]] against the registered `>` [[claim value=0.2 ev=ev:426a8cbed7f6d6dfb61d8fe3ed5264a4 field=thresholds.H-credulity tol=4e-06]].
- Kill K-forensic FIRED on `skepticism_effect` = [[claim value=0.0232563 ev=ev:426a8cbed7f6d6dfb61d8fe3ed5264a4 field=metrics.skepticism_effect tol=1e-06]] (`<` [[claim value=0.1 ev=ev:426a8cbed7f6d6dfb61d8fe3ed5264a4 field=kill_thresholds.K-forensic tol=2e-06]]): the disclosure-game framing adds nothing over raw reliance; the shipped S15 kill

### ATLAS-VCE: Value-convergence excess across the fleet beats the capability-matched null

This card was frozen under an explicit low-power acceptance: power 0.48 at MDE 0.00087371 (recorded in the freeze manifest); a confirmation between the frozen threshold and the MDE is a low-power detection, not a large-effect one.

- H-vce (refuted): value-convergence excess on real reward-model pairs beats the capability-matched random-utility null; `real_vce` = [[claim value=-0.0698739 ev=ev:0c80d9086ebb0ad9704b88a8eefb01fa field=metrics.real_vce tol=1.4e-06]] against the registered `>` [[claim value=0 ev=ev:0c80d9086ebb0ad9704b88a8eefb01fa field=thresholds.H-vce tol=1e-06]].
- H-vce-p (confirmed): the convergent-pair alignment exceeds the random-utility null at p < 0.05; `reward_convergent_p_value` = [[claim value=0.001 ev=ev:0c80d9086ebb0ad9704b88a8eefb01fa field=metrics.reward_convergent_p_value tol=1e-06]] against the registered `<` [[claim value=0.05 ev=ev:0c80d9086ebb0ad9704b88a8eefb01fa field=thresholds.H-vce-p tol=1e-06]].
- Kill K-vce FIRED on `real_vce` = [[claim value=-0.0698739 ev=ev:0c80d9086ebb0ad9704b88a8eefb01fa field=metrics.real_vce tol=1.4e-06]] (`<=` [[claim value=0 ev=ev:0c80d9086ebb0ad9704b88a8eefb01fa field=kill_thresholds.K-vce tol=1e-06]]): convergence is fully explained by shared capability structure

### FACT-KUI: The fleet decodes a property it does not price

This card was frozen under an explicit low-power acceptance: power 0.60 at MDE 0.294628 (recorded in the freeze manifest); a confirmation between the frozen threshold and the MDE is a low-power detection, not a large-effect one.

- H-kui-gap (refuted): the fleet-median KUI gap for the star property exceeds its permuted null; `kui_gap_fleet_median` = [[claim value=0.151523 ev=ev:a34316feccee3beaed93c05ce5888567 field=metrics.kui_gap_fleet_median tol=3e-06]] against the registered `>` [[claim value=0.2 ev=ev:a34316feccee3beaed93c05ce5888567 field=thresholds.H-kui-gap tol=4e-06]].
- H-kui-perm (refuted): the gap clears its permutation null; `kui_gap_perm_p` = [[claim value=0.388561 ev=ev:a34316feccee3beaed93c05ce5888567 field=metrics.kui_gap_perm_p tol=7.8e-06]] against the registered `<` [[claim value=0.05 ev=ev:a34316feccee3beaed93c05ce5888567 field=thresholds.H-kui-perm tol=1e-06]].
- H-kui-control (refuted): a priced control property shows no such gap; `kui_control_fleet_median` = [[claim value=0.353553 ev=ev:a34316feccee3beaed93c05ce5888567 field=metrics.kui_control_fleet_median tol=7.1e-06]] against the registered `<` [[claim value=0.1 ev=ev:a34316feccee3beaed93c05ce5888567 field=thresholds.H-kui-control tol=2e-06]].
- Kill K-kui FIRED on `kui_gap_perm_p` = [[claim value=0.388561 ev=ev:a34316feccee3beaed93c05ce5888567 field=metrics.kui_gap_perm_p tol=7.8e-06]] (`>=` [[claim value=0.05 ev=ev:a34316feccee3beaed93c05ce5888567 field=kill_thresholds.K-kui tol=1e-06]]): no reliable decode-not-price gap across the fleet
- Registered companions: `kui_gap` = [[claim value=0.151523 ev=ev:a34316feccee3beaed93c05ce5888567 field=metrics.kui_gap tol=3e-06]].

### CAPACITY-WELCH: Dark reward tracks the criteria-to-dimension ratio; ArmoRM respects the Welch floor

- H-dark-kdeff (refuted): dark reward grows with the criteria-to-effective-dimension ratio; `dark_reward_kdeff_spearman` = [[claim value=-0.406181 ev=ev:f9e6a003a5a1e0aa0fe3ff1eca0d9c7b field=metrics.dark_reward_kdeff_spearman tol=8.1e-06]] against the registered `>` [[claim value=0.6 ev=ev:f9e6a003a5a1e0aa0fe3ff1eca0d9c7b field=thresholds.H-dark-kdeff tol=1.2e-05]].
- H-capacity-perm (refuted): the fleet correlation clears its permutation null; `capacity_perm_p` = [[claim value=0.893411 ev=ev:f9e6a003a5a1e0aa0fe3ff1eca0d9c7b field=metrics.capacity_perm_p tol=1.8e-05]] against the registered `<` [[claim value=0.05 ev=ev:f9e6a003a5a1e0aa0fe3ff1eca0d9c7b field=thresholds.H-capacity-perm tol=1e-06]].
- H-welch (confirmed): the ArmoRM 19-head coherence matrix respects the Welch floor; `welch_floor_min_slack` = [[claim value=0.816398 ev=ev:f9e6a003a5a1e0aa0fe3ff1eca0d9c7b field=metrics.welch_floor_min_slack tol=1.6e-05]] against the registered `>=` [[claim value=0 ev=ev:f9e6a003a5a1e0aa0fe3ff1eca0d9c7b field=thresholds.H-welch tol=1e-06]].

### EVAL-AWARE: The RM knows it is being tested, and steering along that direction inflates reward

- H-aware-probe (confirmed): a benchmark-versus-organic direction is decodable; `real_probe_balanced_acc` = [[claim value=0.79325 ev=ev:54fb272fd011c8ca07a4959f553341ab field=metrics.real_probe_balanced_acc tol=1.6e-05]] against the registered `>` [[claim value=0.6 ev=ev:54fb272fd011c8ca07a4959f553341ab field=thresholds.H-aware-probe tol=1.2e-05]].
- H-aware-steer (refuted): steering along the direction inflates the score; `delta_r_per_steer` = [[claim value=-0.000209509 ev=ev:54fb272fd011c8ca07a4959f553341ab field=metrics.delta_r_per_steer tol=1e-06]] against the registered `>` [[claim value=0.05 ev=ev:54fb272fd011c8ca07a4959f553341ab field=thresholds.H-aware-steer tol=1e-06]].
- H-aware-organic-fpr (refuted): the probe does not spuriously flag organic prompts (equivalence bound, an affirmative null); `organic_fpr` = [[claim value=0.317 ev=ev:54fb272fd011c8ca07a4959f553341ab field=metrics.organic_fpr tol=6.3e-06]] against the registered `<` [[claim value=0.1 ev=ev:54fb272fd011c8ca07a4959f553341ab field=thresholds.H-aware-organic-fpr tol=2e-06]].

### TOPO-HODGE: Intransitive preference mass no scalar reward can represent

- H-intransitive (confirmed): real k-wise preference data carries a computable intransitive fraction; `intransitive_mass` = [[claim value=0.213976 ev=ev:844ad06a5aeb1d89c21c13c19a28cbd2 field=metrics.intransitive_mass tol=4.3e-06]] against the registered `>` [[claim value=0.03 ev=ev:844ad06a5aeb1d89c21c13c19a28cbd2 field=thresholds.H-intransitive tol=1e-06]].
- Registered companions: `gradient_mass` = [[claim value=0.786024 ev=ev:844ad06a5aeb1d89c21c13c19a28cbd2 field=metrics.gradient_mass tol=1.6e-05]]; `curl_mass` = [[claim value=0.213976 ev=ev:844ad06a5aeb1d89c21c13c19a28cbd2 field=metrics.curl_mass tol=4.3e-06]]; `harmonic_mass` = [[claim value=2.42577e-31 ev=ev:844ad06a5aeb1d89c21c13c19a28cbd2 field=metrics.harmonic_mass tol=1e-06]].

### EMB-LORA: Surface biases enter the reward direction before quality features

This card was frozen under an explicit low-power acceptance: power 0.50 at MDE 2.60464 (recorded in the freeze manifest); a confirmation between the frozen threshold and the MDE is a low-power detection, not a large-effect one.

- H-bias-first (refuted): surface biases reach half their final loading earlier than quality features; `real_bias_before_quality` = [[claim value=0.269231 ev=ev:01c81c7185640f9b962b9115f360732e field=metrics.real_bias_before_quality tol=5.4e-06]] against the registered `>` [[claim value=1 ev=ev:01c81c7185640f9b962b9115f360732e field=thresholds.H-bias-first tol=2e-05]].
- H-stabilize (confirmed): the reward direction settles rather than wandering; `w_r_stabilization_monotone` = [[claim value=1 ev=ev:01c81c7185640f9b962b9115f360732e field=metrics.w_r_stabilization_monotone tol=2e-05]] against the registered `==` [[claim value=1 ev=ev:01c81c7185640f9b962b9115f360732e field=thresholds.H-stabilize tol=2e-05]].

### CAL-TRANSFER: Instrument calibration transfers from CPU toy trunks to a real 0.6B trunk

- H-transfer (refuted): instrument scorecard AUCs on a real 0.6B trunk match the CPU-organism AUCs within 0.15; `max_abs_auc_difference` = [[claim value=0.418983 ev=ev:b37f71c015ebfd697d7deb2a7efb5f7e field=metrics.max_abs_auc_difference tol=8.4e-06]] against the registered `<` [[claim value=0.15 ev=ev:b37f71c015ebfd697d7deb2a7efb5f7e field=thresholds.H-transfer tol=3e-06]].
- Kill K-transfer FIRED on `max_abs_auc_difference` = [[claim value=0.418983 ev=ev:b37f71c015ebfd697d7deb2a7efb5f7e field=metrics.max_abs_auc_difference tol=8.4e-06]] (`>=` [[claim value=0.15 ev=ev:b37f71c015ebfd697d7deb2a7efb5f7e field=kill_thresholds.K-transfer tol=3e-06]]): calibration does not transfer toy-to-real; RESULTS opens with this and six cards drop a rung
- Registered companions: `per_instrument_auc_differences` = [[claim value=0.294026 ev=ev:b37f71c015ebfd697d7deb2a7efb5f7e field=metrics.per_instrument_auc_differences tol=5.9e-06]].

### ADJ-AVP: Attribution versus patching, adjudicated against planted ground truth

- H-avp (refuted): patching's ranking matches the organism's marker-responsive components (the twin-activation-delta answer key) better than attribution's; `recovery_gap` = [[claim value=-0.566038 ev=ev:a34457e83cd4eec10d080804a2533e82 field=metrics.recovery_gap tol=1.1e-05]] against the registered `>` [[claim value=0 ev=ev:a34457e83cd4eec10d080804a2533e82 field=thresholds.H-avp tol=1e-06]].
- Kill K-avp FIRED on `recovery_gap_ci_low` = [[claim value=-0.773585 ev=ev:a34457e83cd4eec10d080804a2533e82 field=metrics.recovery_gap_ci_low tol=1.5e-05]] (`<=` [[claim value=0 ev=ev:a34457e83cd4eec10d080804a2533e82 field=kill_thresholds.K-avp tol=1e-06]]): the two instruments are statistically tied against the marker-response key on the organism; neither localization can be preferred, undercutting every attribution-based number (a noisy key, visible in the emitted margin distribution, is the disclosed alternative reading)
- Registered companions: `patching_recovery_auc` = [[claim value=0.433962 ev=ev:a34457e83cd4eec10d080804a2533e82 field=metrics.patching_recovery_auc tol=8.7e-06]]; `attribution_recovery_auc` = [[claim value=1 ev=ev:a34457e83cd4eec10d080804a2533e82 field=metrics.attribution_recovery_auc tol=2e-05]]; `real8b_avp_spearman` = [[claim value=-0.373581 ev=ev:a34457e83cd4eec10d080804a2533e82 field=metrics.real8b_avp_spearman tol=7.5e-06]].

### CONF-PARTIAL: Do the indices carry information beyond size, accuracy, and lineage

- H-partial-perm (confirmed): the surviving index clears its permutation null; `best_index_perm_p` = [[claim value=0.0467953 ev=ev:491f7f378d0b58dfdadebc9aaee7c983 field=metrics.best_index_perm_p tol=1e-06]] against the registered `<` [[claim value=0.05 ev=ev:491f7f378d0b58dfdadebc9aaee7c983 field=thresholds.H-partial-perm tol=1e-06]].
- H-partial (confirmed): at least one load-bearing index retains a partial correlation above 0.3 after the size-and-accuracy control; `max_abs_partial_corr` = [[claim value=0.648485 ev=ev:491f7f378d0b58dfdadebc9aaee7c983 field=metrics.max_abs_partial_corr tol=1.3e-05]] against the registered `>` [[claim value=0.3 ev=ev:491f7f378d0b58dfdadebc9aaee7c983 field=thresholds.H-partial tol=6e-06]].
- H-partial-lineage (refuted): the surviving index also survives the within-lineage check; `lineage_check_pass` = [[claim value=0 ev=ev:491f7f378d0b58dfdadebc9aaee7c983 field=metrics.lineage_check_pass tol=1e-06]] against the registered `==` [[claim value=1 ev=ev:491f7f378d0b58dfdadebc9aaee7c983 field=thresholds.H-partial-lineage tol=2e-05]].

### T3-DECOMP: A legibility frontier leaves a tacit residual

- H-tacit (confirmed): a residual no short rubric captures; `real_tacit_fraction` = [[claim value=0.972287 ev=ev:a7ed2d6b81c5f79458dad733defb3803 field=metrics.real_tacit_fraction tol=1.9e-05]] against the registered `>` [[claim value=0 ev=ev:a7ed2d6b81c5f79458dad733defb3803 field=thresholds.H-tacit tol=1e-06]].

### T3-FIELD: The reward Hessian's flat subspace overlaps the flagged hackable direction

Inconclusive: missing intermediate: no intermediate 'campaign.subspace.flat' for roster_key='skywork-v2-llama31-8b' slice='rb2-full'; the arc that produces it has not run or its shard was not merged.

### META-LEDGER: The campaign scores its own calibration

- H-brier (refuted): the campaign's directional predictions, each assigned a pre-run probability, beat a coin; `brier_directional` = [[claim value=0.26 ev=ev:3990df7af1488c08ad62786d879b0b0f field=metrics.brier_directional tol=5.2e-06]] against the registered `<` [[claim value=0.25 ev=ev:3990df7af1488c08ad62786d879b0b0f field=thresholds.H-brier tol=5e-06]].
- H-coverage (confirmed): the nominal-0.8 prediction intervals cover the observed values near their nominal rate; `interval_coverage_0p8` = [[claim value=0.75 ev=ev:3990df7af1488c08ad62786d879b0b0f field=metrics.interval_coverage_0p8 tol=1.5e-05]] against the registered `>=` [[claim value=0.6 ev=ev:3990df7af1488c08ad62786d879b0b0f field=thresholds.H-coverage tol=1.2e-05]].
- Kill K-meta FIRED on `brier_directional` = [[claim value=0.26 ev=ev:3990df7af1488c08ad62786d879b0b0f field=metrics.brier_directional tol=5.2e-06]] (`>=` [[claim value=0.25 ev=ev:3990df7af1488c08ad62786d879b0b0f field=kill_thresholds.K-meta tol=5e-06]]): the campaign's directional calls do not beat a coin; the discipline does not calibrate itself
- Registered companions: `n_directional` = [[claim value=16 ev=ev:3990df7af1488c08ad62786d879b0b0f field=metrics.n_directional tol=0.00032]]; `n_intervals` = [[claim value=4 ev=ev:3990df7af1488c08ad62786d879b0b0f field=metrics.n_intervals tol=8e-05]].

## What was dropped, what was admitted, and why

- Checkpoint `tier1` was not evaluated at close-out: total metered spend $17.73 is past the pilot phase (over twice the $2.10 pilot line), so the pilot-phase rule does not apply at close-out (metered sum $17.73).
- Admitted at the Tier 3 gate (`tier3`, metered sum $17.73): T3-DECOMP, T3-27B, T3-FIELD.

## Caveats

### Standing caveats, present in every outcome

What this campaign, even fully confirmed, would not establish. It would not establish that internal measurements forecast outcomes on models larger than the roster's biggest seat, on closed models, or on architectures outside the decoder-only reward-model family the library loads. It would not establish causation: a confirmed forecast is evidence that an idealization holds on real models, not proof that the mechanism is the one the theory names, and the campaign's interventions probe that question without settling it. It would not establish that best-of-n selection behaves like reinforcement learning at deployment KL budgets; the campaign measures best-of-n and infers about RL, and that inference is an assumption, stated here. It would not establish that calibration which transfers to the small real trunk transfers to larger ones; one gap was tested, not every gap. And it would not establish that a reward model's internals forecast the behavior of a policy actually optimized against it in production, because no policy is trained here. A fully confirmed ledger is a beginning that earns the next experiment, not a settled science, and a refuted one is the same discipline reporting the other answer.

- WildChat is the freshness and comparability arm, not an independence control: its prompts post-date the roster's training cutoffs but come from the same pipeline that produced RewardBench 2's prompts, so predictions holding there show distribution-matched generalization, not freedom from contamination. The off-pipeline arena slice carries the independence check.
- Every ArmoRM behavioral number uses the model's native gating-head output; the per-criterion readouts and the legacy row-mean composite appear only in internal geometry analyses and are never quoted as ArmoRM's score.
- Every KL value attached to best-of-n selection is the Beirami bound, reported as a bound on the divergence, not the classical log-n identity.
- Multiplicity control is per registered p-value within a card (Benjamini-Hochberg at 0.05), and every card carries at most one registered p-value, so the within-card correction reduces to the raw threshold; the CI-based confirmatory calls carry no additional multiplicity correction beyond their registered exclusion bounds. The overlay exists and would bind on any future multi-p card; on this registry it is disclosed as inert rather than presented as an active correction.

### This run

- 9 works-ledger feature rows are NOT-RUN; the works-ledger states each blocking reason (a NOT-RUN stretch or gated row does not by itself make the run partial).
- 4 works-ledger feature rows FAILED; each is a filed bug or a documented limitation, never a blank.
- the merged store carries no FreezeManifest evidence row; the freeze-before-run citation cannot be made from the published store and rests on the frozen-dir records alone

## Figures

- `cal-transfer` (light/dark): Calibration transfer failed its frozen equivalence bound: max per-instrument AUC difference 0.419 against the 0.15 bound.
- `card-adj-avp` (light/dark): ADJ-AVP fired its kill criterion (K-avp); the registered negative result.
- `card-atlas-vce` (light/dark): ATLAS-VCE fired its kill criterion (K-vce); the registered negative result.
- `card-cal-transfer` (light/dark): CAL-TRANSFER fired its kill criterion (K-transfer); the registered negative result.
- `card-capacity-welch` (light/dark): CAPACITY-WELCH against its frozen predictions: mixed.
- `card-chi-drift` (light/dark): CHI-DRIFT against its frozen predictions: confirmed.
- `card-conf-partial` (light/dark): CONF-PARTIAL against its frozen predictions: mixed.
- `card-emb-lora` (light/dark): EMB-LORA against its frozen predictions: mixed.
- `card-err-rb2` (light/dark): ERR-RB2 fired its kill criterion (K-err); the registered negative result.
- `card-eval-aware` (light/dark): EVAL-AWARE against its frozen predictions: mixed.
- `card-fact-kui` (light/dark): FACT-KUI fired its kill criterion (K-kui); the registered negative result.
- `card-forecast-cat` (light/dark): FORECAST-CAT against its frozen predictions: mixed.
- `card-forensic-receipt` (light/dark): FORENSIC-RECEIPT fired its kill criterion (K-forensic); the registered negative result.
- `card-gauge-e19` (light/dark): GAUGE-E19 was not adjudicated this run: resolve_subjects failed: PermissionError(13, 'Permission denied').
- `card-gauge-xfam` (light/dark): GAUGE-XFAM was not adjudicated this run: resolve_subjects failed: PermissionError(13, 'Permission denied').
- `card-hack-fore` (light/dark): HACK-FORE was not adjudicated this run: missing intermediate: no intermediate 'campaign.index.table' for roster_key='armorm' slice='diagnostic-v3'; the arc that produces it has not run or its shard was not merged.
- `card-hump` (light/dark): HUMP was not adjudicated this run: missing intermediate: no intermediate 'campaign.recorder.drift' for roster_key='policy-qwen25-1.5b' slice='hump-prompts'; the arc that produces it has not run or its shard was not merged.
- `card-judge-vbc` (light/dark): JUDGE-VBC against its frozen predictions: confirmed.
- `card-ladder` (light/dark): LADDER against its frozen predictions: mixed.
- `card-meta-ledger` (light/dark): META-LEDGER fired its kill criterion (K-meta); the registered negative result.
- `card-ppe-bon` (light/dark): PPE-BON was not adjudicated this run: missing intermediate: no intermediate 'campaign.scores' for roster_key='skywork-v2-qwen3-8b' slice='ppe-best-of-k::gpqa'; the arc that produces it has not run or its shard was not merged.
- `card-style-rmb` (light/dark): STYLE-RMB was not adjudicated this run: missing intermediate: no intermediate 'campaign.bias.battery' for roster_key='armorm' slice='diagnostic-v3-degradation'; the arc that produces it has not run or its shard was not merged.
- `card-surgery` (light/dark): SURGERY against its frozen predictions: mixed.
- `card-t3-decomp` (light/dark): T3-DECOMP against its frozen predictions: confirmed.
- `card-t3-field` (light/dark): T3-FIELD was not adjudicated this run: missing intermediate: no intermediate 'campaign.subspace.flat' for roster_key='skywork-v2-llama31-8b' slice='rb2-full'; the arc that produces it has not run or its shard was not merged.
- `card-topo-hodge` (light/dark): TOPO-HODGE against its frozen predictions: confirmed.
- `card-values-contest` (light/dark): VALUES-CONTEST was not adjudicated this run: missing intermediate: no intermediate 'campaign.scores' for roster_key='armorm' slice='helpsteer3-pref-2000'; the arc that produces it has not run or its shard was not merged.
- `card-verif-prm` (light/dark): VERIF-PRM fired its kill criterion (K-verif); the registered negative result.
- `chidrift-rank` (light/dark): The fluctuation-dissipation prediction on real banks: weaker-model rho 0.964 against the shipped synthetic calibration marked on the panel.
- `hero1-ledger` (light/dark): One row per headline call: right of the zero line means the frozen call held (less-than calls are sign-flipped, p-valued calls plot log10(threshold/p), equality calls can only ever land on the line itself, and interval rows shade (0, 1], the entire held region: 0 is the frozen band's edge, 1 is its dead center); colors carry the final adjudication, so a positive-margin marker downgraded by its CI or FDR overlay stays refuted with the drawn interval showing why; 7 held, 10 refuted, 0 inconclusive, counted from the rows drawn.
- `ladder-extrapolation` (light/dark): Internal quantities extrapolated across a same-recipe ladder; 3 of 3 held-out 8B values landed inside intervals frozen before the 8B was measured.

## What the claims gate covers

The claims gate verifies every claim tag (the double-bracketed value/ev/field/tol markers throughout this document) and every bare `ev:` reference against the evidence store; it does not parse caption prose. The document therefore holds every remaining number to a stated rule: counts are derived from the adjudication records and say so where they appear; dollar figures are metered sums from the spend ledger and are labeled as such; frozen thresholds and bounds render as claim tags against the adjudication record that transcribes the registry (itself spec-hash-verified at adjudication time); and figure captions quote only values that also appear as tagged claims here, beside registered constants labeled as registered.

Every tagged number above is verified against the evidence store by `reward-lens claims`; the build fails if any claim drifts from its evidence.
