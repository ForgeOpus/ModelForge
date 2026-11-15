import { render, screen } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AppProviders, AppRoutes } from "@/App";
import { APP_SHELL_ROUTES } from "@/components/AppShell";

const SCREEN: Record<string, { title: string; marker: string }> = {
  "/app": { title: "Configure a fine-tune", marker: "Goal" },
  "/app/run": { title: "Training run", marker: "No runs yet" },
  "/app/history": { title: "Run history", marker: "Nothing has run yet" },
  "/app/models": { title: "Artifacts", marker: "No trained models yet" },
  "/app/playground": { title: "Playground", marker: "Nothing to talk to yet" },
  "/app/system": { title: "Hardware", marker: "Accelerators" },
  "/app/docs": { title: "Documentation", marker: "Concepts" },
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function installApi() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/system")) {
        return json({
          name: "ModelForge",
          package: "modelforge-finetuning",
          version: "4.0.0",
          providers: [],
          strategies: [],
          use_cases: [],
        });
      }
      if (path.includes("/api/hardware")) {
        return json({
          physical_core_count: 8,
          disk_storage_gb: 100,
          available_disk_storage_gb: 40,
          ram_gb: 32,
          gpus: [],
        });
      }
      if (path.includes("/api/runs")) return json({ runs: [] });
      if (path.includes("/api/models")) return json({ models: [] });
      if (path.includes("/api/playground/status")) {
        return json({ available: true, reason: null, session_id: null });
      }
      return json({ detail: "not found" }, 404);
    }),
  );
}

function LocationEcho() {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}</div>;
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AppProviders>
        <LocationEcho />
        <AppRoutes />
      </AppProviders>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  installApi();
  localStorage.clear();
});

describe("production routes", () => {
  it("renders the public landing page", async () => {
    renderAt("/");
    expect(
      await screen.findByRole("heading", { name: "Fine-tune a model on the GPU you already have." }),
    ).toBeTruthy();
    expect(screen.queryByRole("navigation", { name: "Main" })).toBeNull();
  });

  it("renders every AppShell workbench screen", async () => {
    expect(APP_SHELL_ROUTES).toEqual(Object.keys(SCREEN));
    for (const path of APP_SHELL_ROUTES) {
      const { title, marker } = SCREEN[path];
      const view = renderAt(path);
      expect(await screen.findByRole("heading", { name: title })).toBeTruthy();
      expect(await screen.findByText(marker)).toBeTruthy();
      expect(screen.getByRole("navigation", { name: "Main" })).toBeTruthy();
      view.unmount();
    }
  });

  it("renders a specific run on the active-run screen", async () => {
    renderAt("/app/run/12");
    expect(await screen.findByRole("heading", { name: "Training run" })).toBeTruthy();
    expect(await screen.findByText("No runs yet")).toBeTruthy();
    expect(screen.getByTestId("location").textContent).toBe("/app/run/12");
  });

  it("keeps older workbench URLs on the redesigned screens", async () => {
    const cases: [string, string, string][] = [
      ["/app/artifacts", "/app/models", "No trained models yet"],
      ["/app/settings", "/app/system", "Accelerators"],
      ["/app/live", "/app/run", "No runs yet"],
      ["/app/runs/4", "/app/run/4", "No runs yet"],
      ["/app/history/4", "/app/run/4", "No runs yet"],
    ];
    for (const [from, to, marker] of cases) {
      const view = renderAt(from);
      expect(await screen.findByText(marker)).toBeTruthy();
      expect(screen.getByTestId("location").textContent).toBe(to);
      view.unmount();
    }
  });
});
