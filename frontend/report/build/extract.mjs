// Check what Vite built, then put it where the wheel ships it.
//
// vite-plugin-singlefile inlines everything into dist/index.html; that file is the proof that
// nothing was left outside the bundle. The two files the Python renderer inlines are the same
// chunk and the same stylesheet, kept separate so a reviewer can read them.
//
// Destination: src/reward_lens/render/report/assets/, or RL_REPORT_ASSETS_DIR when a test wants
// to rebuild into a tmpdir and diff without touching the tree.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const frontend = join(here, "..", "..");
const dist = join(frontend, "dist");
const fallback = join(frontend, "..", "src", "reward_lens", "render", "report", "assets");
const out = process.env.RL_REPORT_ASSETS_DIR || fallback;

const js = readFileSync(join(dist, "report.js"), "utf8");
const css = readFileSync(join(dist, "report.css"), "utf8");
const html = readFileSync(join(dist, "index.html"), "utf8");

const fail = (message) => {
  console.error(`build:report refused: ${message}`);
  process.exit(1);
};

// A file:// document has an opaque origin. Everything below is unavailable to it, so its presence
// in the bundle means the report would silently lose a part of itself on the machine that matters.
for (const [needle, why] of [
  ["fetch(", "fetch is blocked from an opaque origin"],
  ["XMLHttpRequest", "no request of any kind can leave the page"],
  ["new Worker", "a worker must be same-origin with its document"],
  ["importScripts", "no second script is loadable"],
  ["WebAssembly", "streaming instantiation needs fetch"],
]) {
  if (js.includes(needle)) fail(`report.js contains ${needle}: ${why}`);
}
// A URL in the bundle is a resource the page cannot reach. Two shapes are inert text rather than
// a resource: the XML namespace names React passes to createElementNS, and the react.dev address
// React prints inside a minified error message. Nothing else may name a host.
const NAMESPACE =
  /https?:\/\/(www\.w3\.org\/[A-Za-z0-9\/.-]*|react\.dev\/errors\/)/g;
const host = js.replace(NAMESPACE, "").match(/https?:\/\/[^\s"'`)]+/);
if (host) fail(`report.js names a host (${host[0].slice(0, 60)}); the page can reach nothing`);
if (/^\s*(import|export)\s/m.test(js)) fail("report.js is a module; the report needs a classic script");
if (/\bimport\s*\(/.test(js)) fail("report.js keeps a dynamic import; nothing may be loaded at runtime");
if (/<(script|link)[^>]+(src|href)=/i.test(html)) fail("dist/index.html still references a file outside itself");
if (css.includes("url(http")) fail("report.css fetches something; no CDN font, no remote asset");

mkdirSync(out, { recursive: true });
writeFileSync(join(out, "report.js"), js);
writeFileSync(join(out, "report.css"), css);
console.log(
  `report.js ${js.length} B, report.css ${css.length} B -> ${out}`,
);
