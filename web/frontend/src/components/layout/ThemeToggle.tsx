import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "../../hooks";
import { Button } from "../ui";

type ThemeChoice = "light" | "system" | "dark";

const NEXT: Record<ThemeChoice, ThemeChoice> = {
  light: "system",
  system: "dark",
  dark: "light",
};

const LABEL: Record<ThemeChoice, string> = {
  light: "Theme: light. Switch to system.",
  system: "Theme: system. Switch to dark.",
  dark: "Theme: dark. Switch to light.",
};

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const current = theme as ThemeChoice;

  return (
    <Button
      variant="ghost"
      size="sm"
      aria-label={LABEL[current]}
      title={LABEL[current]}
      onClick={() => setTheme(NEXT[current])}
    >
      {current === "light" && <Sun size={16} />}
      {current === "system" && <Monitor size={16} />}
      {current === "dark" && <Moon size={16} />}
    </Button>
  );
}
