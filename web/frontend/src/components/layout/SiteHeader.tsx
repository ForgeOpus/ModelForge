import { Link } from "react-router-dom";
import { useMediaQuery } from "../../hooks";
import { Wordmark } from "../brand";
import { ThemeToggle } from "./ThemeToggle";

export function SiteHeader() {
  const showLinks = useMediaQuery("(min-width: 720px)");

  return (
    <header
      style={{
        position: "sticky",
        top: 0,
        zIndex: 50,
        backdropFilter: "blur(10px)",
        WebkitBackdropFilter: "blur(10px)",
        background: "var(--surface-1-blur)",
        borderBottom: "1px solid var(--border)",
      }}
    >
      <div
        className="container"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "var(--space-4)",
          paddingBlock: "var(--space-3)",
        }}
      >
        <Wordmark asLink size="md" markVariant="image" />

        <nav
          className="cluster"
          style={{ gap: "var(--space-4)" }}
          aria-label="Site"
        >
          {showLinks && (
            <>
              <a href="#how-it-works" style={{ color: "var(--text-2)" }}>
                How it decides
              </a>
              <a href="#rationale" style={{ color: "var(--text-2)" }}>
                The rationale
              </a>
            </>
          )}
          <Link
            to="/app"
            className="ui-btn ui-btn--primary ui-btn--sm shine-sweep"
          >
            Open Console
          </Link>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}
