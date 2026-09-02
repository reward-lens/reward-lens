import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./report.css";

// The record is the truth and this page is a rendering of it (D-15). It is read from the JSON
// block, not fetched: a file:// document has an opaque origin and can reach nothing.
const block = document.getElementById("assay");
const host = document.getElementById("rl-report");

if (block && host) {
  const record = JSON.parse(block.textContent ?? "{}");
  const digest = document.body.dataset.bundleManifestDigest ?? null;
  createRoot(host).render(<App record={record} bundleDigest={digest} />);
}
