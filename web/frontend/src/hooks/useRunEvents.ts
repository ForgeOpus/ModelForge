import { useEffect, useMemo, useRef, useState } from "react";
import { eventSocketUrl } from "../api/ws";
import type { LogEvent, TrainingEvent } from "../types/events";
import { isHeartbeat, isLogEvent, isTerminal } from "../types/events";

type Status = "idle" | "connecting" | "open" | "closed" | "error";

const CAP = 500;

export function useRunEvents(runId: number | null): {
  events: TrainingEvent[];
  status: Status;
  lastLog: LogEvent | null;
} {
  const [events, setEvents] = useState<TrainingEvent[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (runId == null) {
      setStatus("idle");
      return;
    }

    setEvents([]);
    let closedByUs = false;
    let retry = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const connect = () => {
      setStatus("connecting");
      const ws = new WebSocket(eventSocketUrl(runId));
      wsRef.current = ws;

      ws.onopen = () => {
        retry = 0;
        setStatus("open");
      };

      ws.onmessage = (msg) => {
        let ev: TrainingEvent;
        try {
          ev = JSON.parse(msg.data as string) as TrainingEvent;
        } catch {
          return;
        }
        if (isHeartbeat(ev)) return;
        setEvents((prev) => {
          const next = [...prev, ev];
          return next.length > CAP ? next.slice(next.length - CAP) : next;
        });
        if (isTerminal(ev.type)) {
          closedByUs = true;
          ws.close();
        }
      };

      ws.onerror = () => setStatus("error");

      ws.onclose = () => {
        setStatus("closed");
        if (!closedByUs && retry < 5) {
          retry++;
          timer = setTimeout(connect, Math.min(1000 * 2 ** retry, 15000));
        }
      };
    };

    connect();

    return () => {
      closedByUs = true;
      clearTimeout(timer);
      const ws = wsRef.current;
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close();
      }
      wsRef.current = null;
    };
  }, [runId]);

  const lastLog = useMemo<LogEvent | null>(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      const ev = events[i];
      if (isLogEvent(ev)) return ev;
    }
    return null;
  }, [events]);

  return { events, status, lastLog };
}
