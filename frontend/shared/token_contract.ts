export const TOKEN_MANIFEST_SCHEMA = "rewardlens-site-token-manifest/1" as const;

export type Digest = `sha256:${string}`;

export type SiteTokenManifest = Readonly<{
  schema: typeof TOKEN_MANIFEST_SCHEMA;
  source: Readonly<{ path: "src/styles/tokens.css"; sha256: Digest }>;
  export: Readonly<{ path: "frontend/tokens/site-tokens.css"; sha256: Digest }>;
  canonical_sha256: Digest;
}>;

export const TOKEN_MANIFEST_IMPORT = "../tokens/site-token-manifest.json" as const;

function canonical(value: unknown): string {
  if (value === null || typeof value === "boolean" || typeof value === "number" || typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (typeof value === "object") {
    const object = value as Record<string, unknown>;
    return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${canonical(object[key])}`).join(",")}}`;
  }
  throw new TypeError("token manifests contain JSON values only");
}

export function canonicalManifestBytes(manifest: SiteTokenManifest): Uint8Array {
  const { canonical_sha256: _digest, ...unsigned } = manifest;
  return new TextEncoder().encode(canonical(unsigned));
}

export async function verifyTokenManifest(manifest: SiteTokenManifest): Promise<void> {
  if (manifest.schema !== TOKEN_MANIFEST_SCHEMA) throw new TypeError("unexpected token manifest schema");
  for (const field of [manifest.source.sha256, manifest.export.sha256, manifest.canonical_sha256]) {
    if (!/^sha256:[0-9a-f]{64}$/.test(field)) throw new TypeError("token manifest digest is not sha256");
  }
  const bytes = canonicalManifestBytes(manifest);
  const actual = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes.buffer as ArrayBuffer)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
  if (`sha256:${actual}` !== manifest.canonical_sha256) throw new TypeError("token manifest canonical digest differs");
}
