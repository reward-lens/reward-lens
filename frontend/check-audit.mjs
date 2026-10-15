#!/usr/bin/env node
// Reviewed build-only advisory disposition. npm's full scanner still runs on every CI build.

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

const fail = (message) => {
  console.error(`frontend audit gate failed: ${message}`);
  process.exit(1);
};
const sha256 = (path) => createHash("sha256").update(readFileSync(path)).digest("hex");
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// This digest fixes the entire dependency graph at the reviewed source-map-js patch.
const pinnedFiles = {
  "package-lock.json": "41da6b795e369f652c6b1267552aacf15b0c4f7bd26ae614c5ae19003f55b33f",
  "vite.config.ts": "e0374230184cc7eb060ac4936122405455e1ac918fa41fe0f0400c356cb5aa65",
  "node_modules/vite-plugin-singlefile/dist/esm/index.js": "548c3fd68a8d981c9c9189104491d6edf008e9ebf7c75365512a1f75409f0785",
};
for (const [path, expected] of Object.entries(pinnedFiles)) {
  if (sha256(path) !== expected) fail(`reviewed file changed: ${path}`);
}

const pkg = JSON.parse(readFileSync("package.json", "utf8"));
const lock = JSON.parse(readFileSync("package-lock.json", "utf8"));
if (pkg.devDependencies?.["vite-plugin-singlefile"] !== "2.3.3") {
  fail("singlefile plugin is no longer the reviewed build-only version");
}
for (const [name, version] of Object.entries({
  braces: "3.0.3",
  micromatch: "4.0.8",
  "vite-plugin-singlefile": "2.3.3",
  "source-map-js": "1.2.2",
})) {
  const node = lock.packages?.[`node_modules/${name}`];
  if (node?.version !== version || node?.dev !== true) {
    fail(`${name} is not the reviewed dev-only dependency version`);
  }
}

const scan = spawnSync("npm", ["audit", "--json"], {
  encoding: "utf8",
  maxBuffer: 16 * 1024 * 1024,
});
if (scan.error || ![0, 1].includes(scan.status)) fail("npm audit did not complete normally");
let report;
try {
  report = JSON.parse(scan.stdout);
} catch {
  fail("npm audit did not return valid JSON");
}
if (report.auditReportVersion !== 2 || !report.vulnerabilities || !report.metadata?.vulnerabilities) {
  fail("npm audit report schema changed");
}
const vulnerabilities = report.vulnerabilities;
const counts = report.metadata.vulnerabilities;
if (scan.status === 0 && Object.keys(vulnerabilities).length === 0 && counts.total === 0) {
  console.log("frontend audit: zero advisories");
  process.exit(0);
}

// GHSA-vfj7-8cjw-p6xm has no patched braces version. In the reviewed frontend,
// the only braces path is plugin -> micromatch -> braces. The plugin is used by
// the Vite build only, with inlinePattern omitted (the plugin's default is []),
// so its guarded micromatch.isMatch calls do not run. The config/plugin hashes
// above bind that applicability finding, and the lock binds the exact packages.
if (scan.status !== 1 || !same(Object.keys(vulnerabilities).sort(), [
  "braces", "micromatch", "vite-plugin-singlefile",
])) fail("new or changed frontend advisory set");
const braces = vulnerabilities.braces;
const micromatch = vulnerabilities.micromatch;
const plugin = vulnerabilities["vite-plugin-singlefile"];
if (
  braces.severity !== "high" || braces.isDirect !== false ||
  !same(braces.nodes, ["node_modules/braces"]) ||
  !same(braces.effects, ["micromatch"]) ||
  braces.via?.length !== 1 ||
  braces.via[0]?.url !== "https://github.com/advisories/GHSA-vfj7-8cjw-p6xm" ||
  braces.via[0]?.range !== "<=3.0.3" ||
  !same(micromatch.via, ["braces"]) ||
  !same(micromatch.effects, ["vite-plugin-singlefile"]) ||
  !same(micromatch.nodes, ["node_modules/micromatch"]) ||
  !same(plugin.via, ["micromatch"]) ||
  !same(plugin.nodes, ["node_modules/vite-plugin-singlefile"]) ||
  plugin.isDirect !== true ||
  ![braces, micromatch, plugin].every((item) => item.severity === "high") ||
  !same([counts.info, counts.low, counts.moderate, counts.high, counts.critical, counts.total], [0, 0, 0, 3, 0, 3])
) fail("reviewed braces advisory identity or dependency path changed");

console.log("frontend audit: only reviewed build-only braces advisory remains; all new advisories fail");
