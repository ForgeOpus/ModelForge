import { Link } from "react-router-dom";
import { BrandMark } from "../brand";

export function SiteFooter() {
  const year = new Date().getFullYear();

  return (
    <footer
      style={{
        borderTop: "1px solid var(--border)",
        color: "var(--text-3)",
        fontSize: "var(--step--1)",
      }}
    >
      <div
        className="container"
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "var(--space-4)",
          paddingBlock: "var(--space-6)",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
          <span style={{ display: "inline-flex", alignItems: "center", gap: "var(--space-2)" }}>
            <BrandMark variant="svg" size={20} />
            <span style={{ color: "var(--text-2)" }}>ModelForge</span>
            <span>&middot; hardware-aware fine-tuning</span>
          </span>
          <span>Runs locally. Your models and datasets never leave the machine.</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-4)" }}>
          <a href="#" style={{ color: "var(--text-3)" }}>
            GitHub
          </a>
          <Link to="/app" style={{ color: "var(--text-3)" }}>
            Open Console
          </Link>
          <span>&copy; {year}</span>
        </div>
      </div>
    </footer>
  );
}
