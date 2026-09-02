import { useEffect, useMemo, useState } from "react";
import {
  absenceOf,
  answer,
  coverage,
  entries,
  entryStatus,
  findingStatus,
  fmt,
  intervalText,
  isEstimate,
  omittedTables,
  scopeLabel,
  spellOut,
  status,
  tableCount,
  type Json,
} from "./model";
import { Chart, Field, Mono, Status, type Mark } from "./ui";
import { cell, columnsOf, hasBlock, readTable, type Row } from "./tables";

export function App({ record, bundleDigest }: { record: Json; bundleDigest: string | null }) {
  const all = useMemo(() => entries(record), [record]);
  const cover = coverage(record);
  const head = answer(record);
  return (
    <div className="report">
      <Header record={record} bundleDigest={bundleDigest} />
      <Answer record={record} head={head} cover={cover} />
      <SubjectAndIntent record={record} />
      <WhatMoved record={record} />
      <Evidence rows={all} />
      <Holes record={record} />
      <Provenance record={record} bundleDigest={bundleDigest} />
    </div>
  );
}

function Header({ record, bundleDigest }: { record: Json; bundleDigest: string | null }) {
  const subject = record.subject ?? {};
  return (
    <header className="head">
      <div className="head-title">
        <span className="tool">reward-lens assay</span>
        <h1>
          {subject.reward_system?.name ?? subject.reward_system?.id ?? "unnamed reward system"}{" "}
          <span className="version">{subject.version?.id}</span>
        </h1>
      </div>
      <dl className="head-meta">
        <Field label="recorded">{record.created}</Field>
        <Field label="producer">
          {record.producer?.tool} {record.producer?.version}
        </Field>
        <Field label="bundle manifest">
          <Mono>{bundleDigest ?? "not bundled"}</Mono>
        </Field>
      </dl>
    </header>
  );
}

function Answer({
  record,
  head,
  cover,
}: {
  record: Json;
  head: ReturnType<typeof answer>;
  cover: ReturnType<typeof coverage>;
}) {
  const blockers = (record.findings ?? []).filter((f: Json) => f.level === "error");
  const tone = blockers.length ? status("fail", "BLOCKING") : status("pass", "NO BLOCKING FINDING");
  return (
    <section className="panel answer" data-section="answer">
      <h2>The answer</h2>
      <p className="sentence">{head.sentence}</p>
      <div className="headline">
        <span className="big">{head.number}</span>
        <span className="caption">{head.caption}</span>
        <Status value={tone} />
      </div>
      <p className="scope" data-scope={record.subject?.version?.id ?? "unknown"}>
        Scope: {scopeLabel(record)}. {cover.absent} of {cover.total} declared measurements are holes,
        and nothing on this page was measured outside that scope.
      </p>
    </section>
  );
}

function SubjectAndIntent({ record }: { record: Json }) {
  const subject = record.subject ?? {};
  const intent = record.intent ?? {};
  const check = intent.outcome_check ?? {};
  const digests = Object.entries(subject.digests ?? {});
  return (
    <section className="panel" data-section="subject">
      <h2>Subject and intent</h2>
      <dl className="grid">
        <Field label="reward system">
          {subject.reward_system?.id} ({subject.reward_system?.name})
        </Field>
        <Field label="version">
          {subject.version?.id} <Mono>{subject.version?.digest}</Mono>
        </Field>
        <Field label="context">
          {subject.context?.configuration ?? "no configuration declared"}
          {subject.context?.policy ? ` · policy ${subject.context.policy}` : ""}
        </Field>
        <Field label="declared success">{intent.success}</Field>
        <Field label="constraints">
          {(intent.constraints ?? []).length ? (
            <ul>
              {(intent.constraints ?? []).map((c: string) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          ) : (
            "none declared"
          )}
        </Field>
        <Field label="outcome check">
          <Status
            value={
              check.state === "qualified"
                ? status("pass", "QUALIFIED")
                : check.state === "unqualified"
                  ? status("warn", "UNQUALIFIED")
                  : status("absent", "NOT MEASURED")
            }
          />{" "}
          {check.kind ? spellOut(check.kind) : "no outcome protocol"}
        </Field>
      </dl>
      <table className="kv">
        <caption>Version digests</caption>
        <thead>
          <tr>
            <th scope="col">component</th>
            <th scope="col">digest</th>
          </tr>
        </thead>
        <tbody>
          {digests.map(([key, value]) => (
            <tr key={key}>
              <th scope="row">{spellOut(key)}</th>
              <td>
                {value ? (
                  <Mono>{String(value)}</Mono>
                ) : (
                  <Status value={status("absent", "NOT MEASURED")} />
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

/**
 * What the record's findings and holes say, and nothing this panel worked out for itself. The
 * per-section executed share that stood here was a rate computed in the browser, which is the one
 * thing D-15 forbids a renderer: every number below is one the record carries.
 */
function WhatMoved({ record }: { record: Json }) {
  const findings = (record.findings ?? []) as Json[];
  const holes = (record.holes ?? []) as Json[];
  const rules = new Map((record.rules ?? []).map((r: Json) => [r.id, r]));
  const ranked = [...findings].sort(
    (a, b) => weight(b.level) - weight(a.level) || String(a.id).localeCompare(String(b.id)),
  );
  return (
    <section className="panel" data-section="findings">
      <h2>What moved the number</h2>
      <p className="holes-named" data-holes={holes.map((h: Json) => h.entry_id).join(" ")}>
        {holes.length === 0
          ? "The record declares no hole, so every entry below rests on a measurement."
          : `Declared and not measured, in the record's own words: ${holes
              .map((h: Json) => `${h.entry_id} (${spellOut(h.state)})`)
              .join("; ")}.`}
      </p>
      {ranked.length === 0 ? (
        <p className="no-finding">
          No blocking finding under this protocol. That is a statement about what was checked, not a
          clean bill: see the holes below.
        </p>
      ) : (
        <ol className="findings">
          {ranked.map((f) => {
            const rule = rules.get(f.rule) as Json | undefined;
            return (
              <li key={f.id} className="finding" data-finding={f.id}>
                <div className="finding-head">
                  <Status value={findingStatus(f)} />
                  <span className="finding-name">{rule?.name ?? f.rule}</span>
                  <Mono>{f.code}</Mono>
                </div>
                <p>{rule?.short_description ?? f.severity_rationale}</p>
                <dl className="grid">
                  <Field label="kind">{spellOut(f.kind)}</Field>
                  <Field label="level">{f.level}</Field>
                  <Field label="scope">{spellOut(f.scope)}</Field>
                  <Field label="arm">{f.arm ? spellOut(f.arm) : "not declared"}</Field>
                  <Field label="witness">
                    {(f.entries ?? []).join(", ") || "no entry named"}
                    {f.location ? ` · ${f.location.file}:${f.location.line}` : ""}
                  </Field>
                  <Field label="why it matters">{f.severity_rationale}</Field>
                </dl>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}

function weight(level: string): number {
  return level === "error" ? 3 : level === "warning" ? 2 : 1;
}

function Evidence({ rows }: { rows: Json[] }) {
  const [query, setQuery] = useState("");
  const needle = query.trim().toLowerCase();
  const shown = needle
    ? rows.filter((e) =>
        `${e.entry_id} ${e.section} ${e.kind} ${e.measurand} ${e.method?.id}`
          .toLowerCase()
          .includes(needle),
      )
    : rows;
  const charts = chartsByUnit(rows);
  return (
    <section className="panel" data-section="evidence">
      <h2>The evidence</h2>
      <label className="filter">
        <span>Filter</span>
        <input
          type="search"
          value={query}
          placeholder="entry, section, method"
          onChange={(e) => setQuery(e.target.value)}
        />
        <span className="count">
          {shown.length} of {rows.length}
        </span>
      </label>
      <div className="scroller">
        <table className="evidence">
          <caption>Every declared measurement, with its method and its interval</caption>
          <thead>
            <tr>
              <th scope="col">entry</th>
              <th scope="col">status</th>
              <th scope="col">measurand</th>
              <th scope="col">value</th>
              <th scope="col">interval</th>
              <th scope="col">method</th>
              <th scope="col">scope</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((e) => {
              const st = entryStatus(e);
              const estimate = isEstimate(e);
              return (
                <tr
                  key={e.entry_id}
                  data-entry={e.entry_id}
                  data-state={e.state}
                  data-tone={st.tone}
                  {...(estimate ? { "data-estimate": e.entry_id } : {})}
                >
                  <th scope="row">
                    <Mono>{e.entry_id}</Mono>
                    <span className="kind">{spellOut(e.kind)}</span>
                  </th>
                  <td>
                    <Status value={st} />
                  </td>
                  <td>
                    {e.measurand}
                    {(e.limitations ?? []).length ? (
                      <ul className="limits">
                        {(e.limitations ?? []).map((l: string) => (
                          <li key={l}>{l}</li>
                        ))}
                      </ul>
                    ) : null}
                  </td>
                  <td>
                    {typeof e.value === "number" ? (
                      <span data-value={String(e.value)}>
                        {fmt(e.value)}
                        {e.unit ? <span className="unit"> {e.unit}</span> : null}
                        {e.n != null ? <span className="n"> n={e.n}</span> : null}
                      </span>
                    ) : (
                      <Absent entry={e} word={st.word} />
                    )}
                  </td>
                  <td>{estimate ? <span className="interval">{intervalText(e)}</span> : "—"}</td>
                  <td>
                    <Mono>{e.method?.id}</Mono>
                    <span className="procedure">{e.method?.procedure}</span>
                  </td>
                  <td>{spellOut(e.scope ?? "")}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {charts.map(([unit, marks]) => (
        <Chart
          key={unit}
          id={`estimates-${slug(unit)}`}
          title={`Point estimates in ${unit}, with their intervals`}
          marks={marks}
          unit={unit}
        />
      ))}
    </section>
  );
}

/**
 * A value the record does not carry. It is the entry's own word and the access that was missing,
 * never a zero: `Number(e.value ?? 0)` used to put an unmeasured entry on the chart at 0, which
 * reads as a measurement of nothing (review answer 4, section 0.4).
 */
function Absent({ entry, word }: { entry: Json; word: string }) {
  const absence = absenceOf(entry);
  return (
    <span className="no-value" data-absent={absence?.state ?? word}>
      {word}
      {absence?.missing_access ? (
        <span className="missing"> missing access: {absence.missing_access}</span>
      ) : null}
    </span>
  );
}

function slug(unit: string): string {
  return unit.replace(/[^A-Za-z0-9]+/g, "-").replace(/^-|-$/g, "").toLowerCase() || "unit";
}

/**
 * One chart per unit, because a share and a second do not share an axis. Only entries that carry
 * a value are drawn; an absence is in the table above, with what was missing, and is drawn
 * nowhere. An estimate with no interval (D-75) keeps its point and says why there is no bar.
 */
function chartsByUnit(rows: Json[]): [string, Mark[]][] {
  const groups = new Map<string, Mark[]>();
  for (const e of rows) {
    if (!isEstimate(e) || typeof e.value !== "number") continue;
    const unit = String(e.unit ?? "no unit declared");
    const marks = groups.get(unit) ?? [];
    marks.push({
      label: e.entry_id,
      value: e.value,
      low: e.uncertainty?.interval?.[0] ?? null,
      high: e.uncertainty?.interval?.[1] ?? null,
      tone: entryStatus(e).tone,
      note: intervalText(e),
    });
    groups.set(unit, marks);
  }
  return [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0]));
}

function Holes({ record }: { record: Json }) {
  const holes = (record.holes ?? []) as Json[];
  return (
    <section className="panel" data-section="holes">
      <h2>The holes</h2>
      {holes.length === 0 ? (
        <p>Every declared measurement was executed.</p>
      ) : (
        <ul className="holes">
          {holes.map((h) => (
            <li key={h.entry_id} data-hole={h.entry_id}>
              <div className="hole-head">
                <Status value={status("absent", spellOut(h.state).toUpperCase())} />
                <Mono>{h.entry_id}</Mono>
              </div>
              <dl className="grid">
                <Field label="missing access">{h.missing_access}</Field>
                <Field label="claims it blocks">
                  <ul>
                    {(h.affected_claims ?? []).map((c: string) => (
                      <li key={c}>{c}</li>
                    ))}
                  </ul>
                </Field>
                <Field label="remedy">
                  <Mono>{h.remedy}</Mono>
                </Field>
              </dl>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function where(id: string, omitted: string[]): string {
  if (omitted.includes(id)) return "the bundle";
  return hasBlock(id) ? "in this file, as a Parquet block" : "referenced, not inlined";
}

// Tier B reads its rows here, out of the block in this same file (D-15). Nothing is fetched: the
// read is asynchronous only because hyparquet's reader is, not because anything leaves the page.
function TableRows({ id, declared }: { id: string; declared: number }) {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    readTable(id)
      .then((read) => {
        if (live && read) setRows(read);
      })
      .catch((error: unknown) => {
        if (live) setFailed(String(error));
      });
    return () => {
      live = false;
    };
  }, [id]);
  if (failed !== null) {
    return (
      <p className="note" data-table-error={id}>
        the Parquet block for {id} did not read: {failed}
      </p>
    );
  }
  if (rows === null) {
    return (
      <p className="note" data-table-pending={id}>
        reading the Parquet block for {id}
      </p>
    );
  }
  const columns = columnsOf(rows);
  return (
    <table className="kv" data-table-rows={id} data-rows={rows.length}>
      <caption>
        {id}: {rows.length} of {declared} rows, read from the Parquet block in this file
      </caption>
      <thead>
        <tr>
          {columns.map((c) => (
            <th key={c} scope="col">
              {c}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i}>
            {columns.map((c) => (
              <td key={c}>{cell(row[c])}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Provenance({ record, bundleDigest }: { record: Json; bundleDigest: string | null }) {
  const p = record.provenance ?? {};
  const cost = record.cost ?? {};
  const env = record.environment_excluded_from_digest ?? {};
  const omitted = omittedTables(record);
  return (
    <section className="panel" data-section="provenance">
      <h2>Provenance</h2>
      <dl className="grid">
        <Field label="offline">{p.offline ? "yes, no egress was available" : "no"}</Field>
        <Field label="sandbox tier">{p.sandbox_tier}</Field>
        <Field label="operating system">{p.os}</Field>
        <Field label="seed">{p.seed}</Field>
        <Field label="access level">{p.access_level ? spellOut(p.access_level) : "not declared"}</Field>
        <Field label="cost">
          {cost.wall_s}s wall, {cost.cpu_s}s cpu, ${cost.usd}, {cost.api_calls} paid calls
        </Field>
        <Field label="schema">{record.$schema}</Field>
        <Field label="embedding tier">
          {record.embedding?.tier}
          {omitted.length
            ? ` · ${omitted.length} of ${tableCount(record)} tables are in the bundle`
            : " · the whole record is on this page"}
        </Field>
        <Field label="bundle manifest">
          <Mono>{bundleDigest ?? "no bundle digest was supplied"}</Mono>
        </Field>
        <Field label="machine">
          {[env.host, env.python, env.cwd].filter(Boolean).join(" · ")}
        </Field>
      </dl>
      <h3>Reproduce</h3>
      <ol className="reproduce">
        {(p.reproduce ?? []).map((c: string) => (
          <li key={c}>
            <Mono>{c}</Mono>
          </li>
        ))}
      </ol>
      <h3>Tables</h3>
      <table className="kv">
        <thead>
          <tr>
            <th scope="col">table</th>
            <th scope="col">rows</th>
            <th scope="col">digest</th>
            <th scope="col">where</th>
          </tr>
        </thead>
        <tbody>
          {((record.tables ?? []) as Json[]).map((t) => (
            <tr key={t.id}>
              <th scope="row">{t.id}</th>
              <td>{t.rows}</td>
              <td>
                <Mono>{t.digest}</Mono>
              </td>
              <td>{where(t.id, omitted)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {((record.tables ?? []) as Json[])
        .filter((t) => hasBlock(String(t.id)))
        .map((t) => (
          <TableRows key={String(t.id)} id={String(t.id)} declared={Number(t.rows ?? 0)} />
        ))}
      <p className="note">
        This page is derived from the record embedded in it and is never referenced by it (D-16).
        The record is the citable artifact.
      </p>
    </section>
  );
}
