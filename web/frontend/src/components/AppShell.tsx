import { NavLink, Outlet, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "motion/react";
import * as Tooltip from "@radix-ui/react-tooltip";
import {
  Activity, Boxes, Cpu, FileText, History, type LucideIcon,
  MessagesSquare, Moon, PanelLeftClose, PanelLeftOpen, SlidersHorizontal, Sun,
} from "lucide-react";
import { useEffect, useState } from "react";
import { Lockup, Mark } from "./Logo";
import { tierColor } from "@/lib/tiers";
import { EASE, TIMING, routeTransition } from "@/lib/motion";
import { useTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";
import { gpuVram, rawVram } from "@/lib/vram";
import type { HardwareSpecs, RunRecord } from "@/types";

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
}

const GROUPS: { heading: string; items: NavItem[] }[] = [
  {
    heading: "Train",
    items: [
      { to: "/app", label: "Configure", icon: SlidersHorizontal },
      { to: "/app/run", label: "Run", icon: Activity },
      { to: "/app/history", label: "History", icon: History },
    ],
  },
  {
    heading: "Models",
    items: [
      { to: "/app/models", label: "Artifacts", icon: Boxes },
      { to: "/app/playground", label: "Playground", icon: MessagesSquare },
    ],
  },
  {
    heading: "System",
    items: [
      { to: "/app/system", label: "Hardware", icon: Cpu },
      { to: "/app/docs", label: "Docs", icon: FileText },
    ],
  },
];

/** Every path the rail links to. The production route map must cover these. */
export const APP_SHELL_ROUTES = GROUPS.flatMap((group) => group.items.map((item) => item.to));

const TITLES: Record<string, string> = {
  "/app": "Configure a fine-tune",
  "/app/run": "Training run",
  "/app/history": "Run history",
  "/app/models": "Artifacts",
  "/app/playground": "Playground",
  "/app/system": "Hardware",
  "/app/docs": "Documentation",
};

export function AppShell({
  hardware,
  activeRun,
}: {
  hardware: HardwareSpecs | null;
  activeRun: RunRecord | null;
}) {
  const { theme, toggle } = useTheme();
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem("modelforge_rail") === "collapsed",
  );

  // Below the md breakpoint a 212px rail leaves too little for the content and
  // the page scrolls sideways, so the rail always runs icon-only there. The
  // stored preference is left untouched for when the window widens again.
  const [narrow, setNarrow] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(max-width: 767px)").matches,
  );

  useEffect(() => {
    const query = window.matchMedia("(max-width: 767px)");
    const update = (e: MediaQueryList | MediaQueryListEvent) => setNarrow(e.matches);
    update(query);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    localStorage.setItem("modelforge_rail", collapsed ? "collapsed" : "open");
  }, [collapsed]);

  const railCollapsed = collapsed || narrow;

  const title =
    TITLES[location.pathname] ??
    (location.pathname.startsWith("/app/run/") ? "Training run" : "ModelForge");

  const totalVram = rawVram(hardware);
  const primaryGpu = hardware?.gpus[0];
  const unifiedMemory = primaryGpu != null && gpuVram(primaryGpu) === null;
  const training = activeRun?.status === "running";

  return (
    <Tooltip.Provider delayDuration={350}>
      <div
        className="grid min-h-screen"
        style={{ gridTemplateColumns: `${railCollapsed ? "60px" : "212px"} minmax(0,1fr)` }}
      >
        <nav
          aria-label="Main"
          className="flex flex-col border-r border-line bg-ground px-2.5 pb-2.5 pt-3.5"
        >
          <div className={cn("mb-4 px-1.5", railCollapsed && "flex justify-center px-0")}>
            {railCollapsed ? <Mark className="size-6 text-brand" /> : <Lockup size="sm" duo />}
          </div>

          {GROUPS.map((group) => (
            <div key={group.heading} className="mb-4">
              {!railCollapsed && (
                <h2 className="eyebrow px-2 pb-1.5 text-[10px]">{group.heading}</h2>
              )}
              {group.items.map((item) => (
                <RailLink
                  key={item.to}
                  item={item}
                  collapsed={railCollapsed}
                  badge={
                    item.to === "/app/run" && activeRun?.status === "running"
                      ? "live"
                      : undefined
                  }
                />
              ))}
            </div>
          ))}

          <div className="mt-auto border-t border-line px-1.5 pt-3">
            {hardware && primaryGpu ? (
              railCollapsed ? (
                <span
                  className={cn("mx-auto block size-1.5 rounded-full", training && "animate-pulse")}
                  style={{ background: training ? tierColor(activeRun?.peft) : "var(--ok)" }}
                  title={training ? "Training" : "Idle"}
                />
              ) : (
                <>
                  <div className="flex items-center gap-1.5 text-[11.5px] text-dim">
                    <span
                      className={cn("size-1.5 shrink-0 rounded-full", training && "animate-pulse")}
                      style={{ background: training ? tierColor(activeRun?.peft) : "var(--ok)" }}
                    />
                    <span className="truncate">{primaryGpu.name}</span>
                  </div>
                  {/* Only draw a bar when it means something. Idle, an empty
                      gauge just looks broken; we report free capacity instead. */}
                  {training && (
                    <div className="mt-1.5 h-[3px] overflow-hidden rounded-full bg-track">
                      <div
                        className="h-full w-1/3 animate-pulse rounded-full"
                        style={{ background: tierColor(activeRun?.peft) }}
                      />
                    </div>
                  )}
                  <div className="mt-1.5 flex justify-between font-mono text-[10.5px] text-faint">
                    <span>
                      {unifiedMemory ? "unified memory" : `${totalVram.toFixed(1)} GB free`}
                    </span>
                    <span>
                      {training
                        ? "training"
                        : hardware.gpus.length > 1
                          ? `${hardware.gpus.length} GPUs`
                          : "idle"}
                    </span>
                  </div>
                </>
              )
            ) : (
              !railCollapsed && <p className="text-[11px] text-faint">No GPU detected</p>
            )}

            <button
              type="button"
              onClick={() => setCollapsed((c) => !c)}
              hidden={narrow}
              className="mt-3 flex w-full items-center gap-2 rounded-xs px-1 py-1 text-[11.5px] text-faint hover:text-text"
              aria-label={railCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {railCollapsed ? (
                <PanelLeftOpen className="size-3.5" />
              ) : (
                <>
                  <PanelLeftClose className="size-3.5" /> Collapse
                </>
              )}
            </button>
          </div>
        </nav>

        <div className="flex min-w-0 flex-col">
          <header className="flex h-13 flex-none items-center gap-3.5 border-b border-line bg-surface px-5">
            <h1 className="font-display text-[15px] font-semibold">{title}</h1>
            <div className="flex-1" />
            <AnimatePresence>
              {activeRun && activeRun.status === "running" && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.94, y: -3 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.94 }}
                  transition={{ duration: TIMING.ui, ease: EASE }}
                >
                  <NavLink
                    to="/app/run"
                    className="flex items-center gap-2 rounded-full border border-line bg-raised py-1 pl-2.5 pr-3 text-[12px] text-dim hover:border-line-strong hover:text-text"
                  >
                    <motion.span
                      className="size-1.5 rounded-full bg-brand"
                      animate={{ opacity: [1, 0.25, 1] }}
                      transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
                    />
                    <b className="font-mono text-[11.5px] font-normal text-text">
                      #{activeRun.id}
                    </b>
                    running
                  </NavLink>
                </motion.div>
              )}
            </AnimatePresence>
            <button
              type="button"
              onClick={toggle}
              aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
              className="rounded-xs p-1.5 text-faint hover:text-text"
            >
              {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
            </button>
          </header>
          <main className="min-h-0 min-w-0 flex-1">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={location.pathname} className="h-full min-w-0" {...routeTransition}>
                <Outlet />
              </motion.div>
            </AnimatePresence>
          </main>
        </div>
      </div>
    </Tooltip.Provider>
  );
}

function RailLink({
  item,
  collapsed,
  badge,
}: {
  item: NavItem;
  collapsed: boolean;
  badge?: string;
}) {
  const Icon = item.icon;
  const link = (
    <NavLink
      to={item.to}
      end={item.to === "/app"}
      className={({ isActive }) =>
        cn(
          "relative flex w-full items-center gap-2.5 rounded-xs px-2 py-1.5 text-[13px] transition-colors duration-150",
          collapsed && "justify-center px-0",
          isActive ? "font-medium text-text" : "text-dim hover:text-text",
        )
      }
    >
      {({ isActive }) => (
        <>
          {isActive && (
            <motion.span
              layoutId="rail-active"
              className="absolute inset-0 rounded-xs bg-raised"
              transition={{ type: "spring", stiffness: 520, damping: 42, mass: 0.7 }}
            />
          )}
          <Icon className="relative z-10 size-3.5 shrink-0 opacity-85" />
          {!collapsed && <span className="relative z-10 truncate">{item.label}</span>}
          {!collapsed && badge && (
            <span className="relative z-10 ml-auto font-mono text-[10px] text-brand">{badge}</span>
          )}
          {collapsed && <span className="sr-only">{item.label}</span>}
        </>
      )}
    </NavLink>
  );

  if (!collapsed) return link;
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>{link}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          side="right"
          sideOffset={8}
          className="z-50 rounded-sm border border-line bg-surface px-2 py-1 text-[12px] text-text shadow-[var(--shadow-card)]"
        >
          {item.label}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}
