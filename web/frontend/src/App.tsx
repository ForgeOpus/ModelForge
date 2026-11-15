import type { ReactNode } from "react";
import { BrowserRouter, Navigate, Route, Routes, useParams } from "react-router-dom";
import { Toaster } from "sonner";

import { AppShell } from "@/components/AppShell";
import { AuthGate } from "@/components/AuthGate";
import { Landing } from "@/landing/Landing";
import { StoreProvider, useStore } from "@/lib/store";
import { ThemeProvider, useTheme } from "@/lib/theme";
import { Configure } from "@/screens/Configure";
import { Docs } from "@/screens/Docs";
import { History } from "@/screens/History";
import { Models } from "@/screens/Models";
import { Playground } from "@/screens/Playground";
import { Run } from "@/screens/Run";
import { System } from "@/screens/System";

function Notifications() {
  const { theme } = useTheme();
  return <Toaster theme={theme} position="bottom-right" closeButton />;
}

function WorkbenchLayout() {
  const { hardware, activeRun, unauthorised } = useStore();
  if (unauthorised) return <AuthGate />;
  return <AppShell hardware={hardware} activeRun={activeRun} />;
}

/** Bookmarks from the previous workbench still open the same run. */
function LegacyRunRedirect() {
  const { runId } = useParams();
  return <Navigate to={runId ? `/app/run/${runId}` : "/app/run"} replace />;
}

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/app" element={<WorkbenchLayout />}>
        <Route index element={<Configure />} />
        <Route path="run" element={<Run />} />
        <Route path="run/:runId" element={<Run />} />
        <Route path="history" element={<History />} />
        <Route path="models" element={<Models />} />
        <Route path="playground" element={<Playground />} />
        <Route path="system" element={<System />} />
        <Route path="docs" element={<Docs />} />
        <Route path="artifacts" element={<Navigate to="/app/models" replace />} />
        <Route path="settings" element={<Navigate to="/app/system" replace />} />
        <Route path="live" element={<Navigate to="/app/run" replace />} />
        <Route path="runs/:runId" element={<LegacyRunRedirect />} />
        <Route path="history/:runId" element={<LegacyRunRedirect />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
      <Route path="*" element={<Landing />} />
    </Routes>
  );
}

export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider>
      <StoreProvider>
        {children}
        <Notifications />
      </StoreProvider>
    </ThemeProvider>
  );
}

/** Single production application: public landing plus the redesigned workbench. */
export function App() {
  return (
    <BrowserRouter>
      <AppProviders>
        <AppRoutes />
      </AppProviders>
    </BrowserRouter>
  );
}
