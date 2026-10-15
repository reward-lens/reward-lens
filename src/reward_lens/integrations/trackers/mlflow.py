"""The optional MLflow SDK through its local file-store tracking APIs."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from ._local import file_report_parent
from .base import Availability, RecordSummary, TrackerRef


class MlflowTracker:
    name = "mlflow"

    def _sdk(self):
        try:
            return importlib.import_module("mlflow")
        except ModuleNotFoundError:
            return None

    def _offline_root(self, report_url: str) -> Path:
        configured = os.environ.get("REWARD_LENS_MLFLOW_DIR")
        if configured:
            return Path(configured)
        return file_report_parent(report_url) / "mlruns"

    def available(self) -> Availability:
        if self._sdk() is not None:
            return Availability.available()
        return Availability.unavailable(
            "reward-lens[trackers]",
            "install reward-lens[trackers] to use the MLflow local SDK backend",
        )

    def write(self, summary: RecordSummary, report_url: str) -> TrackerRef:
        mlflow = self._sdk()
        if mlflow is None:
            raise RuntimeError("MLflow backend was called while unavailable")
        root = self._offline_root(report_url)
        root.mkdir(parents=True, exist_ok=True)
        # MLflow 3.16 retains the local FileStore as an explicit opt-in.  This is the documented
        # offline store for this backend; no server or account is contacted.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        report = file_report_parent(report_url) / Path(report_url).name
        payload = {"summary": summary.to_dict(), "report_url": report_url}
        mlflow.set_tracking_uri(root.resolve().as_uri())
        mlflow.set_experiment("reward-lens")
        with mlflow.start_run(run_name=summary.assay_id):
            mlflow.log_dict(payload, "reward-lens/summary.json")
            mlflow.log_artifact(str(report), artifact_path="reward-lens")
            uri = mlflow.get_artifact_uri("reward-lens/summary.json")
        readback = mlflow.artifacts.load_dict(uri)
        parsed = urlparse(uri)
        path = Path(unquote(parsed.path))
        return TrackerRef(self.name, report_url, path, readback)


def backend() -> MlflowTracker:
    return MlflowTracker()
