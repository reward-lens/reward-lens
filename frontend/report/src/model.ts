// Readers over the record. Nothing here computes a statistic: every number on the page was
// measured by the instrument that wrote the record (D-15). Counting entries is presentation;
// deriving a rate, an interval or an estimate is not, and does not happen in this file.

export type Json = Record<string, any>;

export const SECTIONS = [
  "validity",
  "soundness",
  "reach",
  "exploits",
  "framing",
  "reward_statistics",
  "signal",
  "trace",
  "forecast",
  "calibration",
] as const;

export type SectionName = (typeof SECTIONS)[number];
export type Tone = "pass" | "fail" | "warn" | "absent" | "info";

export interface Status {
  tone: Tone;
  word: string;
  glyph: string;
}

const GLYPH: Record<Tone, string> = {
  pass: "✔",
  fail: "✘",
  warn: "!",
  absent: "○",
  info: "·",
};

export function status(tone: Tone, word: string): Status {
  return { tone, word, glyph: GLYPH[tone] };
}

export function spellOut(value: string): string {
  return value.replace(/_/g, " ");
}

export function entries(record: Json): Json[] {
  const measurement = record.measurement ?? {};
  return SECTIONS.flatMap((s) => (measurement[s] ?? []) as Json[]);
}

/** The status of one entry. An absence carries its own word, and it is never a pass. */
export function entryStatus(entry: Json): Status {
  if (entry.kind === "absence" || entry.state === "absent") {
    return status("absent", spellOut(entry.absence?.state ?? "NOT_MEASURED").toUpperCase());
  }
  if (entry.check) return entry.check.passed ? status("pass", "PASS") : status("fail", "FAIL");
  if (entry.state === "partial") return status("warn", "PARTIAL");
  return status("info", "MEASURED");
}

export function findingStatus(finding: Json): Status {
  if (finding.level === "error") return status("fail", "ERROR");
  if (finding.level === "warning") return status("warn", "WARNING");
  return status("info", "NOTE");
}

export interface Coverage {
  executed: number;
  total: number;
  absent: number;
  partial: number;
}

/** Executed coverage: how many declared entries carry a measurement and how many are holes. */
export function coverage(record: Json): Coverage {
  const all = entries(record);
  const absent = all.filter((e) => e.kind === "absence").length;
  const partial = all.filter((e) => e.state === "partial").length;
  return { executed: all.length - absent, total: all.length, absent, partial };
}

export function blocking(record: Json): Json[] {
  return (record.findings ?? []).filter((f: Json) => f.level === "error");
}

/** The widest scope the page can honestly claim: the one the findings or the entries were run at. */
export function scopeLabel(record: Json): string {
  const fromFindings = blocking(record).map((f) => f.scope);
  const pool = fromFindings.length ? fromFindings : entries(record).map((e) => e.scope);
  const seen = Array.from(new Set(pool.filter(Boolean)));
  if (seen.length === 0) return "no scope was executed";
  return seen.map(spellOut).join(", ");
}

/** The one sentence of section 6.4, in plain language, with the number that carries it. */
export function answer(record: Json): { sentence: string; number: string; caption: string } {
  const blockers = blocking(record);
  const cover = coverage(record);
  if (blockers.length === 0) {
    return {
      sentence: "No blocking finding under this protocol",
      number: `${cover.executed} of ${cover.total}`,
      caption: "declared measurements executed",
    };
  }
  const plural = blockers.length === 1 ? "finding rests" : "findings rest";
  return {
    sentence:
      `${blockers.length} blocking ${plural} on this reward system under this protocol, ` +
      `over ${cover.executed} of ${cover.total} declared measurements`,
    number: String(blockers.length),
    caption: blockers.length === 1 ? "blocking finding" : "blocking findings",
  };
}

/** The absence block of an entry that has none of what it set out to measure. */
export function absenceOf(entry: Json): Json | null {
  return entry.kind === "absence" ? (entry.absence ?? null) : null;
}

export function isEstimate(entry: Json): boolean {
  return entry.kind === "estimate" || typeof entry.value === "number";
}

/** How an interval reads, including the D-75 case where there is honestly no interval. */
export function intervalText(entry: Json): string {
  const u = entry.uncertainty;
  if (!u) return "no interval reported";
  if (!u.interval) {
    const why = u.reason ? `: ${u.reason}` : "";
    const n = u.clusters != null ? ` (${u.clusters} clusters)` : "";
    return `no interval${why}${n}`;
  }
  const level = u.level != null ? `level ${fmt(u.level)}` : "level not stated";
  return `${fmt(u.interval[0])} to ${fmt(u.interval[1])} · ${level} · ${spellOut(u.method)}`;
}

export function fmt(value: unknown): string {
  if (value == null) return "—";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  return String(value);
}

export function omittedTables(record: Json): string[] {
  return (record.embedding?.omitted_tables ?? []) as string[];
}

export function tableCount(record: Json): number {
  return ((record.tables ?? []) as Json[]).length;
}
