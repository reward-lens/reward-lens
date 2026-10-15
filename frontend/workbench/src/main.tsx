import { createRoot } from "react-dom/client";
import { getRecord } from "../../shared/getRecord";
import { WorkbenchState, type ServedState } from "../../shared/workbench_state";
import { CompareRoute } from "../routes/compare";
import { HistoryRoute } from "../routes/history";
import { QueueRoute } from "../routes/queue";
import "../../shared/workbench.css";
import "../../report/src/report.css";

function stateId(): string {
  const match = /^\/state\/([^/]+)$/.exec(window.location.pathname);
  return match?.[1] ?? "case-loaded";
}

function selection(name: "left" | "right" | "rung"): string | null {
  const value = new URLSearchParams(window.location.search).get(name);
  if (!value) return null;
  // The server owns rung semantics and its named invalid-rung refusal. Keep a
  // bounded nonempty query value so that an invalid request reaches that
  // boundary instead of being silently changed into an ambiguous selection.
  if (name === "rung") return value.length <= 32 ? value : null;
  return !value.includes("/") && !value.includes("\\") ? value : null;
}

function routeFor(state: ServedState) {
  const left = selection("left") ?? "demo";
  if (state.screen === "compare") return <CompareRoute selector={left} other={selection("right")} rung={selection("rung")} />;
  if (state.screen === "history") return <HistoryRoute selector={left} />;
  if (state.screen === "queue") return <QueueRoute />;
  return null;
}

async function getState(id: string): Promise<ServedState> {
  const response = await fetch(`/api/states/${encodeURIComponent(id)}`, { credentials: "same-origin" });
  if (!response.ok) throw new Error("The requested workbench state is unavailable.");
  return response.json() as Promise<ServedState>;
}

async function boot() {
  const root = createRoot(document.getElementById("root")!);
  try {
    const [state, load] = await Promise.all([getState(stateId()), getRecord({ kind: "served", id: "demo" })]);
    root.render(<WorkbenchState state={state} load={load} download={() => { window.location.assign("/api/report/demo"); }}>{routeFor(state)}</WorkbenchState>);
  } catch (error) {
    root.render(<main><h1>Workbench state is unavailable</h1><p>{String(error)}</p></main>);
  }
}

void boot();
