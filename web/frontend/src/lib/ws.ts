import { useEffect, useRef, useState } from "react";
import { api, wsUrl } from "./api";
import { normalizeTrainingEvent } from "./progress";
import type { TrainingEvent } from "@/types";

export { deriveProgress } from "./progress";
export type { MetricPoint, RunProgress } from "./progress";

const MAX_EVENTS = 2000;

export type StreamState = "connecting" | "open" | "closed";

/* Live training events for one run.

   Two things this has to get right that the old UI did not:
   the buffer is seeded from the server's replay endpoint, so a page reload
   mid-run repaints the whole history instead of showing an empty log; and the
   socket reconnects with backoff, because a training process that saturates
   the GPU can starve the event loop long enough to drop the connection.
   Replay and live frames are normalized to Unix seconds before they are stored,
   matching workflow.models.TrainingEvent.ts. */
export function useRunEvents(runId: number | null) {
  const [events, setEvents] = useState<TrainingEvent[]>([]);
  const [state, setState] = useState<StreamState>("closed");
  const socketRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);
  const closedByUs = useRef(false);

  useEffect(() => {
    if (runId == null) {
      setEvents([]);
      setState("closed");
      return;
    }

    let cancelled = false;
    closedByUs.current = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const append = (event: TrainingEvent) => {
      if (event.type === "heartbeat") return;
      setEvents((prev) => {
        // Backend timestamps are Unix seconds. Missing ones are filled in that unit once.
        const next = [...prev, normalizeTrainingEvent(event)];
        return next.length > MAX_EVENTS ? next.slice(next.length - MAX_EVENTS) : next;
      });
    };

    const connect = () => {
      if (cancelled) return;
      setState("connecting");
      const socket = new WebSocket(wsUrl(`/api/runs/${runId}/events`));
      socketRef.current = socket;

      socket.onopen = () => {
        if (cancelled) return;
        retryRef.current = 0;
        setState("open");
      };
      socket.onmessage = (message) => {
        try {
          append(JSON.parse(message.data) as TrainingEvent);
        } catch {
          /* a malformed frame should not kill the stream */
        }
      };
      socket.onclose = () => {
        if (cancelled || closedByUs.current) {
          setState("closed");
          return;
        }
        setState("closed");
        retryRef.current += 1;
        const delay = Math.min(1000 * 2 ** (retryRef.current - 1), 15000);
        timer = setTimeout(connect, delay);
      };
      socket.onerror = () => socket.close();
    };

    // Paint history first, then attach the live stream on top of it.
    api
      .runEvents(runId)
      .then((history) => {
        if (cancelled) return;
        setEvents(
          history
            .filter((e) => e.type !== "heartbeat")
            .map((event) => normalizeTrainingEvent(event)),
        );
      })
      .catch(() => {
        /* replay is a convenience; the live stream still works without it */
      })
      .finally(() => {
        if (!cancelled) connect();
      });

    return () => {
      cancelled = true;
      closedByUs.current = true;
      if (timer) clearTimeout(timer);
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [runId]);

  return { events, state };
}
