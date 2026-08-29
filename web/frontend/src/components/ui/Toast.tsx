import {
  createContext,
  useContext,
  useCallback,
  useState,
  type ReactNode,
} from "react";
import { Info, XCircle, CheckCircle2, X } from "lucide-react";
import "./Toast.css";

export interface Toast {
  id: number;
  title: string;
  description?: string;
  tone?: "info" | "danger" | "success";
  action?: { label: string; href: string };
}

interface ToastCtx {
  push: (t: Omit<Toast, "id">) => void;
}

const Ctx = createContext<ToastCtx>({ push: () => {} });

export const useToast = () => useContext(Ctx);

const TONE = {
  info: { color: "var(--info)", icon: <Info size={16} /> },
  danger: { color: "var(--danger)", icon: <XCircle size={16} /> },
  success: { color: "var(--success)", icon: <CheckCircle2 size={16} /> },
} as const;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);

  const dismiss = useCallback((id: number) => {
    setItems((p) => p.filter((x) => x.id !== id));
  }, []);

  const push = useCallback(
    (t: Omit<Toast, "id">) => {
      const id = Date.now() + Math.random();
      setItems((p) => [...p, { ...t, id }]);
      setTimeout(() => dismiss(id), 6000);
    },
    [dismiss],
  );

  return (
    <Ctx.Provider value={{ push }}>
      {children}
      <div className="ui-toast-region" role="region" aria-label="Notifications">
        {items.map((t) => {
          const tone = TONE[t.tone ?? "info"];
          return (
            <div
              key={t.id}
              className="ui-toast fade-in-up"
              role="status"
              style={{ borderLeftColor: tone.color }}
            >
              <span className="ui-toast-icon" style={{ color: tone.color }}>
                {tone.icon}
              </span>
              <div className="ui-toast-content">
                <strong className="ui-toast-title">{t.title}</strong>
                {t.description && <p className="ui-toast-desc">{t.description}</p>}
                {t.action && (
                  <a className="ui-toast-action" href={t.action.href}>
                    {t.action.label}
                  </a>
                )}
              </div>
              <button
                type="button"
                className="ui-toast-close"
                onClick={() => dismiss(t.id)}
                aria-label="Dismiss notification"
              >
                <X size={14} />
              </button>
            </div>
          );
        })}
      </div>
    </Ctx.Provider>
  );
}
