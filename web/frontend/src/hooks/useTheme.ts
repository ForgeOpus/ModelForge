import { useCallback, useEffect, useState } from "react";

type ThemeChoice = "light" | "dark" | "system";
type Resolved = "light" | "dark";

const KEY = "modelforge_theme";

function readStored(): ThemeChoice {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "light" || v === "dark") return v;
  } catch {
    /* ignore */
  }
  return "system";
}

function systemResolved(): Resolved {
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  } catch {
    return "dark";
  }
}

function apply(choice: ThemeChoice): void {
  try {
    const el = document.documentElement;
    if (choice === "system") delete el.dataset.theme;
    else el.dataset.theme = choice;
  } catch {
    /* ignore */
  }
}

export function useTheme(): {
  theme: ThemeChoice;
  resolved: Resolved;
  setTheme: (t: ThemeChoice) => void;
} {
  const [theme, setThemeState] = useState<ThemeChoice>(() => readStored());
  const [resolved, setResolved] = useState<Resolved>(() =>
    readStored() === "system" ? systemResolved() : (readStored() as Resolved),
  );

  useEffect(() => {
    if (theme !== "system") {
      setResolved(theme);
      return;
    }
    setResolved(systemResolved());
    let mql: MediaQueryList;
    try {
      mql = window.matchMedia("(prefers-color-scheme: dark)");
    } catch {
      return;
    }
    const onChange = () => setResolved(mql.matches ? "dark" : "light");
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [theme]);

  const setTheme = useCallback((t: ThemeChoice) => {
    apply(t);
    try {
      if (t === "system") localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, t);
    } catch {
      /* ignore */
    }
    setThemeState(t);
  }, []);

  return { theme, resolved, setTheme };
}
