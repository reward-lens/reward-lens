import { parquetReadObjects } from "hyparquet/src/read.js";

// Tier B's per-row tables (D-15). Each one arrives in the file itself as a base64 Parquet block,
// and hyparquet reads it here. A file:// document has an opaque origin, so there is no fetch, no
// worker and no streaming WebAssembly to reach for; `parquetReadObjects` is imported from its own
// module rather than from the package root so the url helper, the one thing in hyparquet that
// calls fetch, never enters the bundle.

export type Row = Record<string, unknown>;

export function blockId(id: string): string {
  return `table:${id}`;
}

export function hasBlock(id: string): boolean {
  return document.getElementById(blockId(id)) !== null;
}

function blockBytes(id: string): Uint8Array | null {
  const el = document.getElementById(blockId(id));
  if (!el) return null;
  const b64 = (el.textContent ?? "").trim();
  if (!b64) return null;
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

/** The rows of one table, read out of this file. Returns `null` when the file does not carry it. */
export async function readTable(id: string): Promise<Row[] | null> {
  const bytes = blockBytes(id);
  if (!bytes) return null;
  const buffer = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
  const file = {
    byteLength: buffer.byteLength,
    slice: async (start: number, end?: number) => buffer.slice(start, end),
  };
  const rows = (await parquetReadObjects({ file })) as Row[];
  return rows;
}

/** A Parquet value as text. Integers arrive as BigInt and timestamps as Date; neither is a number. */
export function cell(value: unknown): string {
  if (value == null) return "—";
  if (typeof value === "bigint") return value.toString();
  if (value instanceof Date) return value.toISOString();
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function columnsOf(rows: Row[]): string[] {
  const seen: string[] = [];
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!seen.includes(key)) seen.push(key);
    }
  }
  return seen;
}
