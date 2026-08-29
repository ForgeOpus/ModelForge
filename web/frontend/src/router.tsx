import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { ToastProvider } from "./components/ui/Toast";
import { RunActivityProvider } from "./context/RunActivityContext";
import { SiteLayout } from "./components/layout/SiteLayout";
import { AppShell } from "./components/layout/AppShell";
import { LandingPage } from "./pages/landing/LandingPage";
import { NewRunPage } from "./pages/app/NewRunPage";
import { LiveRunPage } from "./pages/app/LiveRunPage";
import { HistoryPage } from "./pages/app/HistoryPage";
import { RunDetailPage } from "./pages/app/RunDetailPage";
import { ArtifactsPage } from "./pages/app/ArtifactsPage";
import { SettingsPage } from "./pages/app/SettingsPage";
import { NotFoundPage } from "./pages/NotFoundPage";

export function AppRouter() {
  return (
    <BrowserRouter>
      <ToastProvider>
        <RunActivityProvider>
          <Routes>
            <Route element={<SiteLayout />}>
              <Route path="/" element={<LandingPage />} />
            </Route>
            <Route path="/app" element={<AppShell />}>
              <Route index element={<NewRunPage />} />
              <Route path="live" element={<LiveRunPage />} />
              <Route path="runs/:runId" element={<LiveRunPage />} />
              <Route path="history" element={<HistoryPage />} />
              <Route path="history/:runId" element={<RunDetailPage />} />
              <Route path="artifacts" element={<ArtifactsPage />} />
              <Route path="settings" element={<SettingsPage />} />
            </Route>
            <Route path="/index.html" element={<Navigate to="/" replace />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </RunActivityProvider>
      </ToastProvider>
    </BrowserRouter>
  );
}
