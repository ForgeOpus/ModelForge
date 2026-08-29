import { resolveToken } from "./client";

/**
 * Build the WebSocket URL for a run's event stream.
 * The backend only accepts the token as a query parameter on the socket.
 */
export function eventSocketUrl(runId: number): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const host = import.meta.env.VITE_WS_HOST ?? window.location.host;
  const token = encodeURIComponent(resolveToken());
  return `${proto}://${host}/api/runs/${runId}/events?token=${token}`;
}
