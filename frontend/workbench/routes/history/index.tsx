import { useEffect, useState } from "react";

type VersionNode = Readonly<{
  assay_id: string;
  created: string;
  version_id: string;
  version_digest: string;
  parents: string[];
  freshness?: "fresh" | "stale" | "unavailable";
  freshness_reason?: string | null;
  rerun_cost?: string | null;
}>;
type HistoryProjection = Readonly<{
  selected: VersionNode;
  graph: Readonly<{ state: "available" | "unavailable"; reason: string | null; remedy: string | null; nodes: VersionNode[] }>;
}>;

async function load(selector: string): Promise<HistoryProjection> {
  const response = await fetch(`/api/history/${encodeURIComponent(selector)}`, { credentials: "same-origin" });
  if (!response.ok) throw new Error("The selected version history is unavailable.");
  return response.json() as Promise<HistoryProjection>;
}

function Freshness({ node }: { node: VersionNode }) {
  if (node.freshness === "fresh") return <p><strong>Evidence is fresh.</strong> {node.freshness_reason}</p>;
  if (node.freshness === "stale") return <p><strong>Evidence is stale.</strong> {node.freshness_reason}</p>;
  return <p><strong>Freshness is not measured.</strong> {node.freshness_reason ?? "No current project version was selected."}</p>;
}

export function HistoryRoute({ selector }: { selector: string }) {
  const [projection, setProjection] = useState<HistoryProjection | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    void load(selector).then(value => { if (live) setProjection(value); }, reason => { if (live) setError(String(reason)); });
    return () => { live = false; };
  }, [selector]);
  if (error) return <section aria-live="polite"><h2>Version history is unavailable</h2><p>{error}</p></section>;
  if (!projection) return <section aria-live="polite"><h2>Loading version history</h2><p>Reading saved project records.</p></section>;
  if (projection.graph.state === "unavailable") return <section className="workbench-route" aria-live="polite"><h2>No version graph</h2><p>{projection.graph.reason}</p><p>{projection.graph.remedy}</p></section>;
  return <section className="workbench-route"><h2>Version history</h2><ol>{projection.graph.nodes.map(node => <li key={node.assay_id}><h3>{node.version_id}</h3><Freshness node={node} /><p>Recorded {node.created}. Rerun cost: {node.rerun_cost ?? "NOT MEASURED"}.</p></li>)}</ol></section>;
}
