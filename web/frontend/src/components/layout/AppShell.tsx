import { useEffect } from "react";
import { Link, Outlet } from "react-router-dom";
import { useRunActivity } from "../../context/RunActivityContext";
import { Sidebar } from "./Sidebar";
import { ThemeToggle } from "./ThemeToggle";
import { TokenBanner } from "./TokenBanner";
import "./AppShell.css";

export function AppShell() {
  const { activeRunId, refreshActivity } = useRunActivity();

  useEffect(() => {
    void refreshActivity();
  }, [refreshActivity]);

  return (
    <div className="app-shell">
      <div className="app-shell__sidebar">
        <Sidebar />
      </div>

      <div className="app-shell__body">
        <div className="app-shell__strip">
          {activeRunId != null && (
            <Link to={`/app/runs/${activeRunId}`} className="app-shell__run-pill">
              <span data-dot aria-hidden="true">
                &#9679;
              </span>
              Run #{activeRunId} live
            </Link>
          )}
          <ThemeToggle />
        </div>

        <div className="container app-shell__main-wrap">
          <TokenBanner />
          <main style={{ paddingBlock: "var(--space-6)" }}>
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  );
}
