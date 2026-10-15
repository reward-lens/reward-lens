---
title: Tracker exports
---

# Tracker exports

`reward-lens export --to BACKEND` sends a summary from the saved assay and the actual local report URI through the tracker registry. The record remains the source of truth.

## file

`file` writes `reward-lens-file-tracker.json` beside the saved report. It contains the assay id, subject-version digest, decision, creation time, producer version, and absolute report URI.

## W&B

`wandb` uses the optional W&B SDK in its real `mode="offline"` path, writes an offline run and
artifact containing the report, and reopens the persisted `wandb-summary.json`. Set
`REWARD_LENS_WANDB_DIR` to choose its local run directory. It never contacts a service or needs an
account in offline mode. If the optional SDK is absent, it refuses with `RL0720` naming
`reward-lens[trackers]`.

## MLflow

`mlflow` uses the optional MLflow SDK with a local file tracking URI, logs the summary with
`log_dict`, attaches the report with `log_artifact`, and reopens the stored summary through
`mlflow.artifacts.load_dict`. Set `REWARD_LENS_MLFLOW_DIR` to choose its local store. If the
optional SDK is absent, it refuses with `RL0720` naming `reward-lens[trackers]`.

## Extending the registry

`registry.discover()` scans tracker modules for a `backend()` factory returning an object with `name`, `available()`, and `write(summary, report_url)`. A later backend joins without a CLI edit.
