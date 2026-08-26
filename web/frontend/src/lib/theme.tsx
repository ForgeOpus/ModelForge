import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

type Theme = "light" | "dark";
const KEY = "modelforge_theme";

const ThemeContext = createContext<{ theme: Theme; toggle: () => void }>({
  theme: "dark",
  toggle: () => {},
});

function initial(): Theme {
  const stored = localStorage.getItem(KEY);
  if (stored === "light" || stored === "dark") return stored;
  // Dark-first: this is tooling that lives next to a terminal, but a system
  // preference for light is still honoured.
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(initial);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    document.documentElement.style.colorScheme = theme;
    localStorage.setItem(KEY, theme);
  }, [theme]);

  return (
    <ThemeContext.Provider
      value={{ theme, toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")) }}
    >
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => useContext(ThemeContext);
