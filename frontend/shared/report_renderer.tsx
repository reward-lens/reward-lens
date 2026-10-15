// The report's own React tree is the shared renderer. The workbench imports
// this boundary instead of maintaining a second record presentation.
import { App } from "../report/src/App";
import type { Json } from "../report/src/model";

export function ReportRenderer({ record }: { record: Json }) {
  return <App record={record} bundleDigest={null} />;
}
