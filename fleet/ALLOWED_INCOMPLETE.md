# Allowed incomplete markers

The ratchet behind clause 5 of the terminal predicate (section 0.1). One line per allowed marker,
`- \`path:line\` | <packet or controller> | <class: abstract|deferred|unclassified> | <reason>`; `path:*` allows a whole file. The list may
shrink and must not grow without a packet id and a reason. Every entry below is the library
baseline at 607774d, recorded by the controller in wave 0 after `fleet/status.py` scanned the tree:
`raise NotImplementedError` in abstract bases and guarded fallbacks that the inherited library
carries. A packet that wires one of these modules into a 4.0 shipped import path either makes the
marker real and deletes the line, or keeps the line with its own packet id and a reason that names
the path and why the raise is correct there.

- `src/reward_lens/concepts/probes.py:23` | controller | abstract | Module docstring: "leaves the gap as a visible TODO on the card" describes shipped behaviour, not a gap
- `src/reward_lens/data/corruptions.py:170` | controller | abstract | corrupt_step docstring TODO(M8+): oracle-matched variants declared out of scope; the mode is guarded
- `src/reward_lens/data/corruptions.py:188` | controller | abstract | DataError message for a non-mechanical mode; guarded refusal listing the built modes as remedy
- `src/reward_lens/data/corruptions.py:387` | controller | abstract | paraphrase_battery docstring noting oracle paraphrases out of scope; the function itself is implemented
- `src/reward_lens/dynamics/checkpoints.py:550` | controller | abstract | from_hf_revisions download gate; the allow_download=True path is implemented at checkpoints.py:598-614
- `src/reward_lens/measure/base.py:504` | controller | abstract | BaseObservable.measure abstract; 80 files under src/ define `def measure(self, ctx`
- `src/reward_lens/measure/composition/composition.py:124` | controller | abstract | CompositionInstrument.compute; overridden by CounterfactualComposition (composition.py:847), abstention.py
- `src/reward_lens/measure/composition/composition.py:128` | controller | abstract | CompositionInstrument.payload; same two concrete subclasses
- `src/reward_lens/measure/composition/tournament.py:767` | controller | abstract | _B6Instrument.payload; overridden at tournament.py:814, 869, 965
- `src/reward_lens/measure/controls/_base.py:43` | controller | abstract | ControlInstrument.compute; overridden in controls/baselines.py, power.py, placebo.py
- `src/reward_lens/measure/controls/_base.py:47` | controller | abstract | ControlInstrument.payload; same three subclasses plus measure/meta/_base.py
- `src/reward_lens/measure/decision/_base.py:98` | controller | abstract | DecisionInstrument.compute; overridden in decision/geometry.py, weights.py, sorting.py, compensation.py
- `src/reward_lens/measure/decision/_base.py:101` | controller | abstract | DecisionInstrument.payload; same four subclass files
- `src/reward_lens/measure/efficiency/cost.py:748` | controller | abstract | _CostInstrument.compute; overridden at cost.py:798, 819
- `src/reward_lens/measure/efficiency/cost.py:774` | controller | abstract | _CostInstrument.payload; overridden at cost.py:855, 878, 902, 927
- `src/reward_lens/measure/efficiency/cost.py:777` | controller | abstract | _CostInstrument.baseline_values; overridden at cost.py:858, 881, 905, 940
- `src/reward_lens/measure/estimator/_base.py:46` | controller | abstract | EstimatorInstrument.compute; six subclass files incl. estimator/mismatch.py, clip.py
- `src/reward_lens/measure/frontier/_base.py:39` | controller | abstract | FrontierInstrument.compute; overridden in frontier/tails.py, surrogate.py, curve.py, horizon.py
- `src/reward_lens/measure/indices/chi.py:899` | controller | abstract | _TripleInstrument.payload; overridden at chi.py:978, 1019, 1041
- `src/reward_lens/measure/indices/heritability.py:699` | controller | abstract | Vestigial _HeritabilityInstrument.payload(): no caller and no override anywhere; see caveat 1
- `src/reward_lens/measure/indices/heritability.py:702` | controller | abstract | _HeritabilityInstrument.compute; overridden at heritability.py:796, 831, 873, 959
- `src/reward_lens/measure/ledger/explained.py:490` | controller | abstract | _ExplainedInstrument.payload; overridden at explained.py:521, 549
- `src/reward_lens/measure/ledger/price.py:1007` | controller | abstract | _LedgerInstrument.payload; overridden at price.py:1072, 1098
- `src/reward_lens/measure/metrology/gstudy.py:119` | controller | abstract | MetrologyInstrument.compute; overridden at gstudy.py:933 and in metrology/grr.py, allocation.py
- `src/reward_lens/measure/metrology/gstudy.py:122` | controller | abstract | MetrologyInstrument.payload; overridden at gstudy.py:1035 and the same two files
- `src/reward_lens/measure/reconcile/instruments.py:263` | controller | abstract | _ReconcileBase.compute; overridden at instruments.py:345, 398, 459
- `src/reward_lens/measure/reconcile/instruments.py:291` | controller | abstract | _ReconcileBase.emit; overridden at instruments.py:348, 401, 492
- `src/reward_lens/measure/threshold/_base.py:138` | controller | abstract | ThresholdInstrument.compute; overridden in threshold/variance.py, bunching.py, deadzone.py, density.py
- `src/reward_lens/monitor/_base.py:116` | controller | abstract | MonitorInstrument.compute; overridden in monitor/cusum.py, eprocess.py, conjunction.py and 2 more
- `src/reward_lens/monitor/_base.py:119` | controller | abstract | MonitorInstrument.payload; same five subclass files
- `src/reward_lens/signals/loaders.py:234` | controller | abstract | Hub-download gate; message states "the code path is implemented; set allow_download=True"
- `src/reward_lens/stats/baselines/base.py:222` | controller | abstract | BaseBaseline.score abstract; overridden in stats/baselines/text.py and series.py
- `src/reward_lens/studies/w6_monitor/_base.py:152` | controller | abstract | W6Instrument.compute; overridden in w6_monitor/behaviour.py and degradation.py
- `src/reward_lens/verifier/mutate.py:814` | controller | abstract | _MutationInstrument._emit; overridden at mutate.py:853, 886

Classes (added 2026-09-13 under the oversight's O-04 distinction): `abstract` is legitimate
non-implementation syntax (an abstract method or protocol body whose concrete subclasses exist and
are what ships); `deferred` is unfinished product behaviour with the packet that will remove it;
`unclassified` is the wave-0 baseline before anyone read the site. `fleet/status.py --strict` fails
while any entry is `deferred` or `unclassified`; classification of the 49 baseline entries is the
controller's at the wave-1 exit, by opening each site.
