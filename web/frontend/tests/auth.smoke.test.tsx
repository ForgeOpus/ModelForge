import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AppProviders, AppRoutes } from "@/App";
import { api, captureStartupToken, getToken } from "@/lib/api";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const SYSTEM = {
  name: "ModelForge",
  package: "modelforge-finetuning",
  version: "4.0.0",
  providers: [],
  strategies: [],
  use_cases: [],
};

const HARDWARE = {
  physical_core_count: 4,
  disk_storage_gb: 20,
  available_disk_storage_gb: 10,
  ram_gb: 16,
  gpus: [],
};

beforeEach(() => {
  localStorage.clear();
  window.history.replaceState({}, "", "/");
  captureStartupToken();
});

describe("startup token", () => {
  it("captures a query token, strips it, and keeps it across refresh and navigation", async () => {
    window.history.replaceState({}, "", "/app?token=startup-secret&stay=1#run");

    expect(captureStartupToken()).toBe("startup-secret");
    expect(window.location.pathname).toBe("/app");
    expect(window.location.search).toBe("?stay=1");
    expect(window.location.hash).toBe("#run");
    expect(localStorage.getItem("modelforge_token")).toBe("startup-secret");

    window.history.replaceState({}, "", "/app/history");
    expect(captureStartupToken()).toBe("startup-secret");
    expect(getToken()).toBe("startup-secret");

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path.includes("/api/system")) return json(SYSTEM);
      if (path.includes("/api/hardware")) return json(HARDWARE);
      if (path.includes("/api/runs")) return json({ runs: [] });
      if (path.includes("/api/health")) return json({ status: "healthy", version: "4.0.0" });
      return json({});
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <MemoryRouter initialEntries={["/app/docs"]}>
        <AppProviders>
          <AppRoutes />
        </AppProviders>
      </MemoryRouter>,
    );
    expect(await screen.findByRole("heading", { name: "Documentation" })).toBeTruthy();

    await api.health();
    const healthCall = fetchMock.mock.calls.find((call) => String(call[0]).includes("/api/health"));
    const headers = (healthCall?.[1] as RequestInit).headers as Record<string, string>;
    expect(headers["X-ModelForge-Token"]).toBe("startup-secret");
  });

  it("accepts a pasted token when the stored one is rejected", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const headers = (init?.headers ?? {}) as Record<string, string>;
      const authorised = headers["X-ModelForge-Token"] === "pasted-token";
      const path = String(input);
      if (!authorised) return json({ detail: "Invalid token." }, 401);
      if (path.includes("/api/system")) return json(SYSTEM);
      if (path.includes("/api/hardware")) return json(HARDWARE);
      if (path.includes("/api/runs")) return json({ runs: [] });
      return json({});
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <MemoryRouter initialEntries={["/app"]}>
        <AppProviders>
          <AppRoutes />
        </AppProviders>
      </MemoryRouter>,
    );

    expect(await screen.findByRole("heading", { name: "This session needs a token" })).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Startup token"), { target: { value: "pasted-token" } });
    fireEvent.submit(screen.getByRole("button", { name: /Connect/ }).closest("form")!);

    expect(await screen.findByRole("heading", { name: "Configure a fine-tune" })).toBeTruthy();
    expect(getToken()).toBe("pasted-token");
  });
});
