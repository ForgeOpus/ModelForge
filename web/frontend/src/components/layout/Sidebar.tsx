import { Activity, Boxes, History, Rocket, Settings } from "lucide-react";
import { NavLink } from "react-router-dom";
import { cn } from "../../lib/cn";
import { useRunActivity } from "../../context/RunActivityContext";
import { Wordmark } from "../brand";
import "./Sidebar.css";

interface NavItem {
  to: string;
  label: string;
  icon: typeof Rocket;
  end?: boolean;
  live?: boolean;
}

const ITEMS: NavItem[] = [
  { to: "/app", label: "New Run", icon: Rocket, end: true },
  { to: "/app/live", label: "Live", icon: Activity, live: true },
  { to: "/app/history", label: "History", icon: History },
  { to: "/app/artifacts", label: "Artifacts", icon: Boxes },
  { to: "/app/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const { activeRunId } = useRunActivity();

  return (
    <div className="sidebar">
      <div className="sidebar__brand">
        <Wordmark asLink size="sm" />
      </div>
      <nav className="sidebar__nav" aria-label="App">
        {ITEMS.map(({ to, label, icon: Icon, end, live }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) => cn("sidebar__link", isActive && "is-active")}
          >
            <Icon size={18} />
            <span className="sidebar__label">{label}</span>
            {live && activeRunId != null && (
              <span className="sidebar__dot" aria-hidden="true" />
            )}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
