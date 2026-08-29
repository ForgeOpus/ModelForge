import { Link } from "react-router-dom";
import { BrandMark } from "../components/brand";

export function NotFoundPage() {
  return (
    <main
      className="container"
      style={{
        minHeight: "60vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        gap: "var(--space-4)",
        paddingBlock: "var(--space-12)",
      }}
    >
      <BrandMark variant="svg" size={56} />
      <h1>Page not found</h1>
      <p className="muted">That route doesn&rsquo;t exist in ModelForge.</p>
      <div className="cluster">
        <Link to="/">Landing</Link>
        <Link to="/app">Open Console</Link>
      </div>
    </main>
  );
}
