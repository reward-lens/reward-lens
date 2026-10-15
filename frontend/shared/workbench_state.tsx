import type { ReactNode } from "react";
import type { RecordLoad } from "./getRecord";
import { ReportRenderer } from "./report_renderer";

export type ServedState = Readonly<{
  id: string;
  screen: "case" | "compare" | "history" | "queue";
  label: string;
  next_action: string;
}>;

export function WorkbenchState({ state, load, download, children }: { state: ServedState; load: RecordLoad; download: () => void; children?: ReactNode }) {
  return <main className="workbench-shell" data-state={state.id}>
    <header className="workbench-state"><p>reward-lens / local workbench</p><p>{state.screen}</p><h1>{state.label}</h1><p>{state.next_action}</p>{state.id === "case-loaded" ? <button onClick={download}>Download report</button> : null}</header>
    {state.id === "case-loaded" && load.ok ? <ReportRenderer record={load.record} /> : null}
    {state.id === "case-loaded" && !load.ok ? <section aria-live="polite"><h2>Record cannot be read</h2><p>{load.message}</p></section> : null}
    {children}
  </main>;
}
