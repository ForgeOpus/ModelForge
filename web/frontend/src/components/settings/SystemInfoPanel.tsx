import { useSystem } from "../../hooks";
import { Spinner, Callout, Stat, StatGrid, Tag, Table, THead, TBody, TR, TH, TD } from "../ui";

export function SystemInfoPanel() {
  const { data, error, loading } = useSystem();

  if (loading) return <Spinner />;
  if (error || !data) return <Callout tone="danger">Could not load system info.</Callout>;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>
      <StatGrid>
        <Stat label="Version" value={data.version} />
        <Stat label="Package" value={data.package} />
      </StatGrid>

      <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
        <span style={{ fontSize: "var(--step--1)", color: "var(--text-3)" }}>Providers</span>
        <div className="cluster">
          {data.providers.map((p) => (
            <Tag key={p}>{p}</Tag>
          ))}
        </div>
      </div>

      <Table>
        <THead>
          <TR>
            <TH>Strategy</TH>
            <TH>Slug</TH>
            <TH>Batch ×</TH>
            <TH>Extra copies</TH>
            <TH>Description</TH>
          </TR>
        </THead>
        <TBody>
          {data.strategies.map((s) => (
            <TR key={s.slug}>
              <TD>{s.name}</TD>
              <TD>
                <Tag>{s.slug}</Tag>
              </TD>
              <TD>{s.batch_multiplier}</TD>
              <TD>{s.extra_model_copies}</TD>
              <TD>{s.description}</TD>
            </TR>
          ))}
        </TBody>
      </Table>
    </div>
  );
}
