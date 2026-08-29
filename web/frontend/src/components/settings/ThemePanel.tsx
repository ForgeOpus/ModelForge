import { useTheme } from "../../hooks";

const OPTIONS: { value: "light" | "system" | "dark"; label: string }[] = [
  { value: "light", label: "Light" },
  { value: "system", label: "System" },
  { value: "dark", label: "Dark" },
];

export function ThemePanel() {
  const { theme, resolved, setTheme } = useTheme();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-3)" }}>
      <div className="cluster" role="radiogroup" aria-label="Theme">
        {OPTIONS.map((opt) => (
          <label
            key={opt.value}
            style={{ display: "inline-flex", alignItems: "center", gap: "var(--space-2)", color: "var(--text-2)" }}
          >
            <input
              type="radio"
              name="theme"
              value={opt.value}
              checked={theme === opt.value}
              onChange={() => setTheme(opt.value)}
            />
            {opt.label}
          </label>
        ))}
      </div>
      <span style={{ fontSize: "var(--step--1)", color: "var(--text-3)" }}>
        Currently showing: {resolved}
      </span>
    </div>
  );
}
