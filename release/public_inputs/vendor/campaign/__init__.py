"""The pre-registered validation campaign for reward-lens 2.0.

This package is the executable form of CAMPAIGN_DESIGN.md: the Tier 0 smoke suite
(``python -m campaign.smoke``), the freeze machinery (``python -m campaign.freeze``), the Modal
app (``campaign/app.py``), one thin analysis module per experiment card, the data converters,
the spend meter, and the reporting layer that closes the campaign out into a scored ledger.

Conventions the whole package holds to:

- Metric names come from ``campaign.registry`` and nowhere else. Specs and analyses both read
  the same CardDef, and T0-PIPE asserts every analysis emits every metric its card registers.
- Analyses are thin: they read ``run.subjects``, call the library's calibrated primitives, and
  emit metrics. They never load models themselves; the arcs do, once per (model, dataset).
- Predictions with ``ci_excludes`` are adjudicated by ``campaign.adjudicate`` on top of the
  shipped runner, which checks comparator and threshold only.
- Tier 0/1 evidence lives in ``runs/pilot`` and is never read by a confirmatory analysis.
- Seeds derive from ``campaign.config.task_seed``; nothing calls the clock or an unseeded RNG
  on an adjudication path.
"""

__all__ = ["registry", "config"]
