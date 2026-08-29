import { useRef } from "react";
import type { ReactNode, KeyboardEvent } from "react";
import "./Tabs.css";

export interface TabItem {
  id: string;
  label: ReactNode;
  icon?: ReactNode;
}

export interface TabsProps {
  tabs: TabItem[];
  active: string;
  onChange: (id: string) => void;
}

export function Tabs({ tabs, active, onChange }: TabsProps) {
  const ref = useRef<HTMLDivElement>(null);

  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    const idx = tabs.findIndex((t) => t.id === active);
    if (idx < 0) return;
    let next = idx;
    if (e.key === "ArrowRight") next = (idx + 1) % tabs.length;
    else if (e.key === "ArrowLeft") next = (idx - 1 + tabs.length) % tabs.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = tabs.length - 1;
    else return;
    e.preventDefault();
    onChange(tabs[next].id);
    const btns = ref.current?.querySelectorAll<HTMLButtonElement>("[role='tab']");
    btns?.[next]?.focus();
  }

  return (
    <div className="ui-tabs" role="tablist" ref={ref} onKeyDown={onKeyDown}>
      {tabs.map((t) => {
        const selected = t.id === active;
        return (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={selected}
            tabIndex={selected ? 0 : -1}
            className={selected ? "ui-tab ui-tab--active" : "ui-tab"}
            onClick={() => onChange(t.id)}
          >
            {t.icon && <span className="ui-tab-icon">{t.icon}</span>}
            {t.label}
          </button>
        );
      })}
    </div>
  );
}
