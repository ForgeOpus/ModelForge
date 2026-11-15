import { useStore } from "@/lib/store";
import { gpuVram, rawVram, adjustedVram, selectTier } from "@/lib/vram";
import { TIER_ORDER, TIERS } from "@/lib/tiers";
import { formatParams } from "@/lib/utils";
import { Alert, Eyebrow, Panel, Skeleton } from "@/components/ui/misc";

/* Everything on this screen used to be a JSON.stringify dump. The data was
   always here; it just wasn't readable. */
export function System() {
  const { system, hardware, loading, error } = useStore();

  if (loading) {
    return (
      <div className="grid gap-4 p-6 md:grid-cols-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-32" />
        ))}
      </div>
    );
  }

  const total = rawVram(hardware);
  const cpuOnly = !hardware || hardware.gpus.length === 0;
  const mps = hardware?.gpus.some((g) => gpuVram(g) === null) ?? false;
  const selection = selectTier(adjustedVram(total), {
    cpuOnly,
    mps,
    ramGb: hardware?.ram_gb ?? 0,
  });

  const diskPct = hardware
    ? ((hardware.disk_storage_gb - hardware.available_disk_storage_gb) / hardware.disk_storage_gb) * 100
    : 0;

  return (
    <div className="flex flex-col gap-6 p-6">
      {error && <Alert tone="danger" title="Hardware detection failed">{error}</Alert>}

      <section>
        <Eyebrow className="mb-3">Accelerators</Eyebrow>
        {cpuOnly ? (
          <Alert tone="warn" title="No GPU detected">
            Training will run on the CPU, which is extremely slow and caps you at very small
            models. Quantisation is unavailable.
          </Alert>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {hardware!.gpus.map((gpu, i) => {
              const vram = gpuVram(gpu);
              return (
                <Panel key={i} className="p-4">
                  <div className="flex items-baseline justify-between gap-3">
                    <h3 className="font-display text-[14px] font-semibold">{gpu.name}</h3>
                    <span className="font-mono text-[11px] text-faint">GPU {i}</span>
                  </div>
                  <p className="mt-2 font-display text-[26px] font-bold leading-none tnum">
                    {vram != null ? vram.toFixed(1) : "—"}
                    <span className="ml-1.5 font-sans text-[12px] font-normal text-faint">
                      {vram != null ? "GB VRAM" : "unified memory"}
                    </span>
                  </p>
                  {vram == null && (
                    <p className="mt-2 text-[11.5px] leading-relaxed text-faint">
                      Apple Silicon shares memory with the system and reports no separate VRAM
                      figure, so the engine plans against a conservative fixed budget.
                    </p>
                  )}
                </Panel>
              );
            })}
          </div>
        )}
      </section>

      <section>
        <Eyebrow className="mb-3">What fits</Eyebrow>
        <Panel className="p-4">
          <p className="mb-3 text-[12.5px] leading-relaxed text-dim">
            Largest model each precision tier can hold, given{" "}
            <span className="font-mono text-text">{total.toFixed(1)} GB</span> of VRAM. The engine
            takes the highest-quality tier that leaves room to train.
          </p>
          <div className="flex flex-col gap-2">
            {TIER_ORDER.map((t) => {
              const ceiling = selection.byTier[t];
              const chosen = selection.tier === t;
              const widest = Math.max(...Object.values(selection.byTier), 1);
              return (
                <div key={t} className="grid grid-cols-[132px_minmax(0,1fr)_82px] items-center gap-3">
                  <span
                    className="font-mono text-[11.5px]"
                    style={{ color: chosen ? TIERS[t].cssVar : "var(--faint)" }}
                  >
                    {t}
                  </span>
                  <span className="h-1.5 overflow-hidden rounded-full bg-track">
                    <span
                      className="block h-full rounded-full"
                      style={{
                        width: `${Math.min(100, (ceiling / widest) * 100)}%`,
                        background: TIERS[t].cssVar,
                        opacity: chosen ? 1 : 0.4,
                      }}
                    />
                  </span>
                  <span className="text-right font-mono text-[11.5px] text-dim tnum">
                    {ceiling > 0 ? formatParams(ceiling) : "—"}
                  </span>
                </div>
              );
            })}
          </div>
        </Panel>
      </section>

      {hardware && (
        <section>
          <Eyebrow className="mb-3">Machine</Eyebrow>
          <div className="grid gap-3 sm:grid-cols-3">
            <Panel className="p-4">
              <p className="text-[11px] uppercase tracking-[0.13em] text-faint">CPU cores</p>
              <p className="mt-1.5 font-display text-[22px] font-bold tnum">
                {hardware.physical_core_count}
              </p>
            </Panel>
            <Panel className="p-4">
              <p className="text-[11px] uppercase tracking-[0.13em] text-faint">System RAM</p>
              <p className="mt-1.5 font-display text-[22px] font-bold tnum">
                {hardware.ram_gb.toFixed(1)}
                <span className="ml-1 font-sans text-[12px] font-normal text-faint">GB</span>
              </p>
            </Panel>
            <Panel className="p-4">
              <p className="text-[11px] uppercase tracking-[0.13em] text-faint">Disk free</p>
              <p className="mt-1.5 font-display text-[22px] font-bold tnum">
                {hardware.available_disk_storage_gb.toFixed(0)}
                <span className="ml-1 font-sans text-[12px] font-normal text-faint">
                  / {hardware.disk_storage_gb.toFixed(0)} GB
                </span>
              </p>
              <span className="mt-2 block h-1.5 overflow-hidden rounded-full bg-track">
                <span
                  className="block h-full rounded-full"
                  style={{
                    width: `${diskPct}%`,
                    background: diskPct > 90 ? "var(--danger)" : "var(--activation)",
                  }}
                />
              </span>
            </Panel>
          </div>
        </section>
      )}

      {system && (
        <section>
          <Eyebrow className="mb-3">Installed</Eyebrow>
          <Panel className="overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-[12.5px]">
                <thead>
                  <tr className="border-b border-line text-left">
                    {["Strategy", "Trains", "Tasks", "Providers"].map((h) => (
                      <th key={h} scope="col" className="px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.13em] text-faint">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {system.strategies.map((s) => (
                    <tr key={s.slug} className="border-b border-line last:border-b-0">
                      <td className="px-3 py-2.5 font-mono">{s.slug}</td>
                      <td className="px-3 py-2.5 text-dim">{s.name}</td>
                      <td className="px-3 py-2.5 font-mono text-faint">
                        {s.supported_tasks.join(", ")}
                      </td>
                      <td className="px-3 py-2.5 font-mono text-faint">
                        {s.supported_providers.join(", ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
          <p className="mt-2.5 font-mono text-[11px] text-faint">
            {system.package} {system.version} · providers: {system.providers.join(", ")}
          </p>
        </section>
      )}
    </div>
  );
}
