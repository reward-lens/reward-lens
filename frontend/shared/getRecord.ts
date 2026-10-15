export type RecordSource =
  | { kind: "inline"; value: string }
  | { kind: "file"; file: File }
  | { kind: "served"; id: string };

export type RecordDocument = Readonly<Record<string, unknown>>;

export type RecordLoad =
  | Readonly<{ ok: true; record: RecordDocument; source: RecordSource["kind"] }>
  | Readonly<{ ok: false; source: RecordSource["kind"]; code: "malformed" | "over-cap" | "unavailable"; message: string }>;

// The browser is intentionally not a schema author. It only establishes that
// it has an object produced by the SDK, then displays its existing fields.
const MAX_ATTACH_BYTES = 26_214_400;

function parsed(raw: string, source: RecordSource["kind"]): RecordLoad {
  try {
    const value: unknown = JSON.parse(raw);
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return { ok: false, source, code: "malformed", message: "The record is not a JSON object." };
    }
    const record = value as Record<string, unknown>;
    if (typeof record.assay_id !== "string" || typeof record.$schema !== "string") {
      return { ok: false, source, code: "malformed", message: "The record lacks its sealed identity or schema." };
    }
    return { ok: true, record, source };
  } catch {
    return { ok: false, source, code: "malformed", message: "The record is not valid JSON." };
  }
}

export async function getRecord(source: RecordSource): Promise<RecordLoad> {
  if (source.kind === "inline") return parsed(source.value, source.kind);
  if (source.kind === "file") {
    if (source.file.size > MAX_ATTACH_BYTES) {
      return { ok: false, source: source.kind, code: "over-cap", message: "This attachment is over the portable report cap. Use the local workbench." };
    }
    return parsed(await source.file.text(), source.kind);
  }
  try {
    const response = await fetch(`/api/record/${encodeURIComponent(source.id)}`, { credentials: "same-origin" });
    if (!response.ok) return { ok: false, source: source.kind, code: "unavailable", message: "The served record is unavailable." };
    return parsed(await response.text(), source.kind);
  } catch {
    return { ok: false, source: source.kind, code: "unavailable", message: "The served record could not be reached." };
  }
}
