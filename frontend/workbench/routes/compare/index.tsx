import { useEffect, useState } from "react";

type RecordSummary = Readonly<{
  assay_id: string;
  created: string;
  version_id: string;
  version_digest: string;
  parents: string[];
}>;

type ScorecardRow = Readonly<{ id: string; label: string; value: string; state: string; remedy: string | null }>;
type Tradeoff = Readonly<{ option_a: string; option_b: string; competing_consequences: string[] }>;
type ScorecardColumn = Readonly<{
  version: RecordSummary;
  rung: string;
  scope: string;
  rows: ScorecardRow[];
  tradeoff: Tradeoff | null;
}>;
type CompareProjection = Readonly<{
  selected: RecordSummary;
  other: RecordSummary;
  scorecard: Readonly<{
    state: "available" | "unavailable";
    reason: string | null;
    remedy: string | null;
    columns?: ScorecardColumn[];
  }>;
}>;

async function load(selector: string, other: string | null, rung: string | null): Promise<CompareProjection> {
  const params = new URLSearchParams();
  if (other) params.set("right", other);
  if (rung) params.set("rung", rung);
  const query = params.size ? `?${params.toString()}` : "";
  const response = await fetch(`/api/compare/${encodeURIComponent(selector)}${query}`, {
    credentials: "same-origin",
  });
  if (!response.ok) throw new Error("The selected comparison records are unavailable.");
  return response.json() as Promise<CompareProjection>;
}

function Version({ record }: { record: RecordSummary }) {
  return <dl className="workbench-route-facts">
    <dt>Version</dt><dd>{record.version_id}</dd>
    <dt>Assay</dt><dd>{record.assay_id}</dd>
    <dt>Recorded</dt><dd>{record.created}</dd>
  </dl>;
}

export function CompareRoute({ selector, other, rung }: { selector: string; other: string | null; rung: string | null }) {
  const [projection, setProjection] = useState<CompareProjection | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    void load(selector, other, rung).then(
      value => { if (live) setProjection(value); },
      reason => { if (live) setError(String(reason)); },
    );
    return () => { live = false; };
  }, [selector, other, rung]);
  if (error) return <section aria-live="polite"><h2>Comparison is unavailable</h2><p>{error}</p></section>;
  if (!projection) return <section aria-live="polite"><h2>Loading comparison</h2><p>Reading the selected sealed records.</p></section>;
  if (projection.scorecard.state === "unavailable") {
    return <section className="workbench-route" aria-live="polite">
      <h2>Common scorecard</h2>
      <p className="workbench-absence">NOT MEASURED</p>
      <p>{projection.scorecard.reason}</p>
      <p>{projection.scorecard.remedy}</p>
      <div className="workbench-route-columns"><Version record={projection.selected} /><Version record={projection.other} /></div>
    </section>;
  }
  const columns = projection.scorecard.columns ?? [];
  return <section className="workbench-route">
    <h2>Common scorecard</h2>
    <div className="workbench-route-columns">
      {columns.map(column => <section key={column.version.assay_id}>
        <h3>{column.version.version_id}</h3>
        <p><strong>Assay</strong> {column.version.assay_id}</p>
        <p><strong>Rung</strong> {column.rung} · <strong>Scope</strong> {column.scope}</p>
        <table><thead><tr><th scope="col">Row</th><th scope="col">Value</th><th scope="col">State</th></tr></thead>
          <tbody>{column.rows.map(row => <tr key={row.id}><th scope="row">{row.label}</th><td>{row.value}{row.remedy ? <><br /><small>Next: {row.remedy}</small></> : null}</td><td>{row.state}</td></tr>)}</tbody>
        </table>
        {column.tradeoff ? <section><h4>Unresolved tradeoff</h4><p>{column.tradeoff.option_a} and {column.tradeoff.option_b} have competing consequences.</p><ul>{column.tradeoff.competing_consequences.map(item => <li key={item}>{item}</li>)}</ul></section> : null}
      </section>)}
    </div>
  </section>;
}
