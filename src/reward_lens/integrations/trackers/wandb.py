"""The optional W&B SDK through its real offline run and artifact APIs."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

from ._local import file_report_parent
from .base import Availability, RecordSummary, TrackerRef


class WandbTracker:
    name = "wandb"

    def _sdk(self):
        try:
            return importlib.import_module("wandb")
        except ModuleNotFoundError:
            return None

    def _offline_root(self, report_url: str) -> Path:
        configured = os.environ.get("REWARD_LENS_WANDB_DIR") or os.environ.get("WANDB_DIR")
        return Path(configured) if configured else file_report_parent(report_url) / "wandb"

    def available(self) -> Availability:
        if self._sdk() is not None:
            return Availability.available()
        return Availability.unavailable(
            "reward-lens[trackers]",
            "install reward-lens[trackers] to use the W&B offline SDK backend",
        )

    def write(self, summary: RecordSummary, report_url: str) -> TrackerRef:
        wandb = self._sdk()
        if wandb is None:
            raise RuntimeError("W&B backend was called while unavailable")
        root = self._offline_root(report_url)
        report = Path(file_report_parent(report_url) / Path(report_url).name)
        payload = {"summary": summary.to_dict(), "report_url": report_url}
        run = wandb.init(
            project="reward-lens", name=summary.assay_id, mode="offline", dir=str(root)
        )
        if run is None:
            raise RuntimeError("W&B did not create an offline run")
        try:
            run.summary.update(payload)
            # W&B 0.30 stores offline summaries in its binary run bundle. This explicit artifact
            # payload is logged through the real run and can be reopened without a W&B service.
            summary_path = Path(run.dir) / "reward-lens-summary.json"
            summary_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            artifact = wandb.Artifact(
                f"reward-lens-{summary.assay_id.split(':', 1)[-1][:16]}",
                type="reward-lens-assay",
                metadata=payload,
            )
            artifact.add_file(str(report), name="report.html")
            artifact.add_file(str(summary_path), name="summary.json")
            run.log_artifact(artifact)
        finally:
            run.finish()
        if not summary_path.is_file():
            raise RuntimeError("W&B offline run did not persist its tracker summary artifact")
        readback = json.loads(summary_path.read_text(encoding="utf-8"))
        return TrackerRef(self.name, report_url, summary_path, readback)


def backend() -> WandbTracker:
    return WandbTracker()
