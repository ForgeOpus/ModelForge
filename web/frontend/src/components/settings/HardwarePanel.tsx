import { useHardware } from "../../hooks";
import { Spinner, Callout, Stat, StatGrid, Table, THead, TBody, TR, TH, TD } from "../ui";
import { formatGb, formatInt } from "../../lib/format";

export function HardwarePanel() {
  const { data, error, loading } = useHardware();

  if (loading) return <Spinner />;
  if (error || !data) return <Callout tone="danger">Could not load hardware specs.</Callout>;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>
      <StatGrid>
        <Stat label="CPU cores" value={formatInt(data.physical_core_count)} />
        <Stat label="RAM" value={formatGb(data.ram_gb)} />
        <Stat label="Disk free" value={formatGb(data.available_disk_storage_gb)} />
        <Stat label="Disk total" value={formatGb(data.disk_storage_gb)} />
      </StatGrid>

      {data.gpus.length === 0 ? (
        <Callout tone="warning">No GPUs detected — training will run on CPU.</Callout>
      ) : (
        <Table>
          <THead>
            <TR>
              <TH>GPU</TH>
              <TH>VRAM</TH>
            </TR>
          </THead>
          <TBody>
            {data.gpus.map((gpu, i) => (
              <TR key={`${gpu.name}-${i}`}>
                <TD>{gpu.name}</TD>
                <TD>
                  {gpu.total_memory_gb === "N/A"
                    ? "Apple MPS — shared"
                    : formatGb(gpu.total_memory_gb)}
                </TD>
              </TR>
            ))}
          </TBody>
        </Table>
      )}
    </div>
  );
}
