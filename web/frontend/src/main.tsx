import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes, useParams } from "react-router-dom";
import { Toaster } from "sonner";

import "./index.css";
import { AppShell } from "@/components/AppShell";
import { AuthGate } from "@/components/AuthGate";
import { ThemeProvider } from "@/lib/theme";
import { StoreProvider, useStore } from "@/lib/store";
import { Landing } from "@/landing/Landing";
import { Configure } from "@/screens/Configure";
import { Docs } from "@/screens/Docs";
import { History } from "@/screens/History";
import { Models } from "@/screens/Models";
import { Playground } from "@/screens/Playground";
import { Run } from "@/screens/Run";
import { System } from "@/screens/System";

/** Keep bookmarks from the v4 preview useful while the route names stay concise. */
function LegacyRunRedirect() {
  const { runId } = useParams();
  return <Navigate replace to={runId ? `/app/run/${runId}` : "/app/run"} />;
}

function Workbench() {
  const { hardware, activeRun, unauthorised } = useStore();

  if (unauthorised) return <AuthGate />;

  return <AppShell hardware={hardware} activeRun={activeRun} />;
}

function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route
            path="/app"
            element={
              <StoreProvider>
                <Workbench />
              </StoreProvider>
            }
          >
            <Route index element={<Configure />} />
            <Route path="run" element={<Run />} />
            <Route path="run/:runId" element={<Run />} />
            <Route path="history" element={<History />} />
            <Route path="models" element={<Models />} />
            <Route path="playground" element={<Playground />} />
            <Route path="system" element={<System />} />
            <Route path="docs" element={<Docs />} />

            <Route path="runs/:runId" element={<LegacyRunRedirect />} />
            <Route path="history/:runId" element={<LegacyRunRedirect />} />
            <Route path="live" element={<Navigate replace to="/app/run" />} />
            <Route path="artifacts" element={<Navigate replace to="/app/models" />} />
            <Route path="settings" element={<Navigate replace to="/app/system" />} />
            <Route path="*" element={<Navigate replace to="/app" />} />
          </Route>
          <Route path="*" element={<Navigate replace to="/" />} />
        </Routes>
        <Toaster richColors theme="system" />
      </BrowserRouter>
    </ThemeProvider>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
