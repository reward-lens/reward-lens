import { useEffect, useState } from "react";

type QueueSnapshot = Readonly<{
  changed: boolean;
  updated_ns: number;
  state: "running" | "idle" | "failed";
  run_id: string | null;
  spent_usd: string | null;
  budget_usd?: string | null;
  next_action: string;
  reason?: string;
}>;

async function load(): Promise<QueueSnapshot> {
  const response = await fetch("/api/queue?since=0", { credentials: "same-origin" });
  if (!response.ok) throw new Error("Queue status is unavailable.");
  return response.json() as Promise<QueueSnapshot>;
}

export function QueueRoute() {
  const [snapshot, setSnapshot] = useState<QueueSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    void load().then(value => { if (live) setSnapshot(value); }, reason => { if (live) setError(String(reason)); });
    return () => { live = false; };
  }, []);
  if (error) return <section aria-live="polite"><h2>Queue status is unavailable</h2><p>{error}</p></section>;
  if (!snapshot) return <section aria-live="polite"><h2>Loading local queue</h2><p>Reading the CLI run directory.</p></section>;
  if (snapshot.state === "idle") return <section className="workbench-route"><h2>No active run</h2><p>{snapshot.next_action}</p></section>;
  if (snapshot.state === "failed") return <section className="workbench-route" aria-live="polite"><h2>Queue status is unavailable</h2><p>{snapshot.reason}</p><p>{snapshot.next_action}</p></section>;
  return <section className="workbench-route"><h2>Run in progress</h2><dl className="workbench-route-facts"><dt>Run</dt><dd>{snapshot.run_id}</dd><dt>Spent</dt><dd>{snapshot.spent_usd ?? "NOT MEASURED"}</dd><dt>Budget</dt><dd>{snapshot.budget_usd ?? "NOT MEASURED"}</dd></dl><p>{snapshot.next_action}</p></section>;
}
