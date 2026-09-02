import type { ReactNode } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { useEffect, useRef } from "react";
import { fmt, type Status as StatusValue } from "./model";

/** A status is a word plus a glyph plus a colour, never colour alone (section 6.4). */
export function Status({ value }: { value: StatusValue }) {
  return (
    <span className="status" data-status={value.word} data-tone={value.tone}>
      <span className="glyph" aria-hidden="true">
        {value.glyph}
      </span>
      <span className="word">{value.word}</span>
    </span>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="field">
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

export function Mono({ children }: { children: ReactNode }) {
  return <span className="mono">{children}</span>;
}

export interface Mark {
  label: string;
  /** The value the record carries, or `null` where it carries none. A null is never drawn as 0. */
  value: number | null;
  low?: number | null;
  high?: number | null;
  tone?: string;
  /** The record's own word for a value that is absent, e.g. `NOT MEASURED` or `REFUSED`. */
  word?: string;
  /** What stands where an interval would, e.g. D-75's `no interval: ... (12 clusters)`. */
  note?: string;
}

/**
 * The drawn range, read off the marks. A chart has to map values to a length, and the only
 * honest map is one whose ends are values the record carries and that are printed beside it;
 * clamping into [0, 1] drew every value above 1 at the same place as 1 (review answer 4).
 */
export function domainOf(marks: Mark[]): [number, number] | null {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const m of marks) {
    for (const v of [m.value, m.low, m.high]) {
      if (v == null || !Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return lo <= hi ? [lo, hi] : null;
}

/**
 * The chart boundary of D-68. The report's own SVG mark set draws the fixed vocabulary; a series
 * dense enough that canvas wins goes to uPlot, which is why uPlot is a dependency here and
 * Observable Plot is not: on file:// there is no compression, so minified bytes are what count.
 */
const DENSE = 300;

export function Chart({
  id,
  title,
  marks,
  unit,
}: {
  id: string;
  title: string;
  marks: Mark[];
  unit: string;
}) {
  const domain = domainOf(marks);
  return (
    <figure className="chart" data-chart={id}>
      <figcaption>{title}</figcaption>
      <p className="domain" data-domain={domain ? `${fmt(domain[0])}:${fmt(domain[1])}` : "none"}>
        {domain
          ? `Drawn from ${fmt(domain[0])} to ${fmt(domain[1])} ${unit}, the smallest and largest this record carries.`
          : `Nothing to draw: no entry here carries a value in ${unit}.`}
      </p>
      {marks.length >= DENSE ? (
        <DenseSeries marks={marks} title={title} />
      ) : (
        <IntervalMarks marks={marks} domain={domain} title={title} />
      )}
      <table className="chart-data">
        <caption>{title}, as a table</caption>
        <thead>
          <tr>
            <th scope="col">item</th>
            <th scope="col">{unit}</th>
            <th scope="col">interval</th>
          </tr>
        </thead>
        <tbody>
          {marks.map((m) => (
            <tr key={m.label} data-row={m.label}>
              <th scope="row">{m.label}</th>
              <td>
                {m.value == null ? (
                  <span className="no-value">{m.word ?? "NOT MEASURED"}</span>
                ) : (
                  <span data-value={String(m.value)}>{fmt(m.value)}</span>
                )}
              </td>
              <td>
                {m.note ??
                  (m.low == null || m.high == null
                    ? "none reported"
                    : `${fmt(m.low)} to ${fmt(m.high)}`)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}

/** A point estimate and its interval, one row per mark. SVG so it prints as vector. */
function IntervalMarks({
  marks,
  domain,
  title,
}: {
  marks: Mark[];
  domain: [number, number] | null;
  title: string;
}) {
  const row = 26;
  const pad = 8;
  const height = marks.length * row + pad * 2;
  const x = (v: number) => {
    if (domain === null) return 50;
    const [lo, hi] = domain;
    if (hi === lo) return 50;
    return 4 + ((v - lo) / (hi - lo)) * 92;
  };
  return (
    <svg
      className="marks"
      viewBox={`0 0 100 ${height}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={title}
    >
      <title>{title}</title>
      {marks.map((m, i) => {
        const y = pad + i * row + row / 2;
        const lo = m.low == null ? null : x(m.low);
        const hi = m.high == null ? null : x(m.high);
        return (
          <g
            key={m.label}
            className={`mark tone-${m.tone ?? "info"}`}
            data-mark={m.label}
            {...(m.value == null ? { "data-undrawn": m.word ?? "NOT MEASURED" } : {})}
          >
            <line className="axis" x1={4} x2={96} y1={y} y2={y} />
            {lo != null && hi != null ? (
              <line className="whisker" x1={lo} x2={hi} y1={y} y2={y} />
            ) : null}
            {m.value == null ? null : (
              <circle className="dot" cx={x(m.value)} cy={y} r={2.4} data-at={String(m.value)} />
            )}
          </g>
        );
      })}
    </svg>
  );
}

function DenseSeries({ marks, title }: { marks: Mark[]; title: string }) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!host.current) return;
    const xs = marks.map((_, i) => i);
    const ys: (number | null)[] = marks.map((m) => m.value);
    const plot = new uPlot(
      {
        title,
        width: host.current.clientWidth || 640,
        height: 220,
        series: [{}, { label: title, stroke: "currentColor" }],
        legend: { show: false },
      },
      [xs, ys],
      host.current,
    );
    return () => plot.destroy();
  }, [marks, title]);
  return <div className="dense" ref={host} />;
}
