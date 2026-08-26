import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Toaster } from "sonner";
import "./index.css";

import { ThemeProvider } from "@/lib/theme";
import { StoreProvider, useStore } from "@/lib/store";
import { AppShell } from "@/components/AppShell";
import { AuthGate } from "@/components/AuthGate";
import { Landing } from "@/landing/Landing";
import { Configure } from "@/screens/Configure";
import { Run } from "@/screens/Run";
import { HistoryScreen } from "@/screens/History";
import { Models } from "@/screens/Models";
import { Playground } from "@/screens/Playground";
import { System } from "@/screens/System";
import { Docs } from "@/screens/Docs";

function Shell() {
  const { hardware, activeRun, unauthorised } = useStore();
  if (unauthorised) return <AuthGate />;
  return <AppShell hardware={hardware} activeRun={activeRun} />;
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/app" element={<Shell />}>
          <Route index element={<Configure />} />
          <Route path="run" element={<Run />} />
          <Route path="run/:runId" element={<Run />} />
          <Route path="history" element={<HistoryScreen />} />
          <Route path="models" element={<Models />} />
          <Route path="playground" element={<Playground />} />
          <Route path="system" element={<System />} />
          <Route path="docs" element={<Docs />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <StoreProvider>
        <App />
        <Toaster
          position="bottom-right"
          toastOptions={{
            style: {
              background: "var(--surface)",
              border: "1px solid var(--line)",
              color: "var(--text)",
              fontSize: "13px",
            },
          }}
        />
      </StoreProvider>
    </ThemeProvider>
  </StrictMode>,
);
