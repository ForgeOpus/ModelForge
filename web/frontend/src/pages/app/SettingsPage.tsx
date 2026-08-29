import { useState } from "react";
import { Cpu, Info, KeyRound, Palette, Server } from "lucide-react";
import { Tabs } from "../../components/ui";
import {
  AboutPanel,
  HardwarePanel,
  SystemInfoPanel,
  ThemePanel,
  TokenPanel,
} from "../../components/settings";

const TABS = [
  { id: "system", label: "System", icon: <Server size={15} /> },
  { id: "hardware", label: "Hardware", icon: <Cpu size={15} /> },
  { id: "token", label: "Token", icon: <KeyRound size={15} /> },
  { id: "theme", label: "Theme", icon: <Palette size={15} /> },
  { id: "about", label: "About", icon: <Info size={15} /> },
];

export function SettingsPage() {
  const [tab, setTab] = useState("system");

  return (
    <div className="stack" style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>
      <h1>Settings</h1>
      <Tabs tabs={TABS} active={tab} onChange={setTab} />
      <div>
        {tab === "system" && <SystemInfoPanel />}
        {tab === "hardware" && <HardwarePanel />}
        {tab === "token" && <TokenPanel />}
        {tab === "theme" && <ThemePanel />}
        {tab === "about" && <AboutPanel />}
      </div>
    </div>
  );
}
