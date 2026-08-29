import { Table, THead, TBody, TR, TH, TD } from "../ui";

export interface DataFormatSpecProps {
  spec: import("../../types/api").DataFormatSpec;
}

export function DataFormatSpec({ spec }: DataFormatSpecProps) {
  const aliases = Object.entries(spec.column_aliases);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
      <p style={{ margin: 0, color: "var(--text-2)", lineHeight: "var(--leading-normal)" }}>
        {spec.user_facing_description}
      </p>
      {aliases.length > 0 && (
        <Table>
          <THead>
            <TR>
              <TH>Alias</TH>
              <TH>Canonical column</TH>
            </TR>
          </THead>
          <TBody>
            {aliases.map(([alias, canonical]) => (
              <TR key={alias}>
                <TD className="mono">{alias}</TD>
                <TD className="mono">{canonical}</TD>
              </TR>
            ))}
          </TBody>
        </Table>
      )}
    </div>
  );
}
