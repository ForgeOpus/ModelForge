import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { Loader2, MessagesSquare, Send, Square } from "lucide-react";

import { api, ApiError, wsUrl } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { SavedModel } from "@/types";

import { Button } from "@/components/ui/button";
import { Label, Textarea } from "@/components/ui/field";
import { Alert, EmptyState, Eyebrow } from "@/components/ui/misc";

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
}

export function Playground() {
  const [models, setModels] = useState<SavedModel[]>([]);
  const [selected, setSelected] = useState<SavedModel | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [blocked, setBlocked] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [systemPrompt, setSystemPrompt] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [maxTokens, setMaxTokens] = useState(256);

  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.listModels().then(setModels).catch(() => setModels([]));
    api.playgroundStatus()
      .then((s) => setBlocked(s.available ? null : s.reason))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  useEffect(() => () => socketRef.current?.close(), []);

  const openSession = useCallback(async (model: SavedModel) => {
    setLoading(true);
    setBlocked(null);
    try {
      const session = await api.openPlayground(model.model_path);
      setSessionId(session.session_id);
      setSelected(model);
      setMessages([]);
      if (session.note) toast.message(session.note);
    } catch (err) {
      const message = err instanceof ApiError ? err.friendly : String(err);
      // A 409 here is the GPU-contention guard, which is a state, not a fault.
      if (err instanceof ApiError && err.status === 409) setBlocked(message);
      else toast.error(message);
    } finally {
      setLoading(false);
    }
  }, []);

  const send = useCallback(() => {
    if (!sessionId || !draft.trim() || streaming) return;

    const outgoing: Message[] = [
      ...(systemPrompt.trim() ? [{ role: "system" as const, content: systemPrompt.trim() }] : []),
      ...messages,
      { role: "user" as const, content: draft.trim() },
    ];

    setMessages((prev) => [...prev, { role: "user", content: draft.trim() }, { role: "assistant", content: "" }]);
    setDraft("");
    setStreaming(true);

    const socket = new WebSocket(wsUrl(`/api/playground/${sessionId}/chat`));
    socketRef.current = socket;

    socket.onopen = () =>
      socket.send(
        JSON.stringify({
          messages: outgoing,
          temperature,
          top_p: topP,
          max_new_tokens: maxTokens,
        }),
      );

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "token") {
        setMessages((prev) => {
          const next = [...prev];
          next[next.length - 1] = {
            role: "assistant",
            content: next[next.length - 1].content + data.text,
          };
          return next;
        });
      } else if (data.type === "done") {
        setStreaming(false);
        socket.close();
      } else if (data.type === "error") {
        toast.error(data.message);
        setStreaming(false);
        socket.close();
      }
    };
    socket.onerror = () => {
      setStreaming(false);
      socket.close();
    };
    socket.onclose = () => setStreaming(false);
  }, [sessionId, draft, streaming, messages, systemPrompt, temperature, topP, maxTokens]);

  function stop() {
    socketRef.current?.send(JSON.stringify({ type: "stop" }));
    setStreaming(false);
  }

  if (blocked) {
    return (
      <div className="p-6">
        <Alert tone="warn" title="The GPU is busy">
          {blocked}
        </Alert>
        <p className="mt-3 text-[12.5px] text-dim">
          <Link to="/app/run" className="underline underline-offset-4 hover:text-text">
            Watch the run
          </Link>{" "}
          — the playground opens as soon as it finishes.
        </p>
      </div>
    );
  }

  if (!sessionId) {
    if (models.length === 0) {
      return (
        <EmptyState
          icon={<MessagesSquare />}
          title="Nothing to talk to yet"
          action={
            <Button variant="primary" asChild>
              <Link to="/app">Configure a fine-tune</Link>
            </Button>
          }
        >
          Once a run finishes, load the trained model here and see how it answers.
        </EmptyState>
      );
    }
    return (
      <div className="flex flex-col gap-3 p-6">
        <Eyebrow>Choose a model to load</Eyebrow>
        <p className="max-w-prose text-[12.5px] leading-relaxed text-dim">
          The model is loaded into GPU memory and stays there until you close it, so the first
          message takes longer than the rest.
        </p>
        {models.map((m) => (
          <button
            key={m.id}
            type="button"
            onClick={() => openSession(m)}
            disabled={loading}
            className="flex items-center gap-3 rounded-md border border-line bg-ground p-3.5 text-left hover:border-line-strong disabled:opacity-60"
          >
            <div className="min-w-0 flex-1">
              <span className="block text-[13px] font-medium">{m.name}</span>
              <span className="mt-0.5 block truncate font-mono text-[11px] text-faint">
                {m.base_model}
              </span>
            </div>
            {loading && <Loader2 className="size-4 animate-spin text-faint" />}
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className="grid h-full grid-cols-1 lg:grid-cols-[minmax(0,1fr)_260px]">
      <div className="flex min-h-0 flex-col">
        <div ref={scrollRef} className="flex-1 overflow-auto px-6 py-5">
          {messages.length === 0 ? (
            <p className="py-10 text-center text-[13px] text-faint">
              Loaded {selected?.name}. Say something to start.
            </p>
          ) : (
            <div className="mx-auto flex max-w-3xl flex-col gap-4">
              {messages.map((m, i) => (
                <div
                  key={i}
                  className={cn(
                    "rounded-md border px-3.5 py-2.5 text-[13px] leading-relaxed",
                    m.role === "user"
                      ? "self-end border-brand/40 bg-raised"
                      : "self-start border-line bg-ground",
                  )}
                  style={{ maxWidth: "88%" }}
                >
                  <span className="mb-1 block text-[10px] uppercase tracking-[0.13em] text-faint">
                    {m.role}
                  </span>
                  <span className="whitespace-pre-wrap">
                    {m.content}
                    {streaming && i === messages.length - 1 && (
                      <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-brand align-middle" />
                    )}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="flex gap-2 border-t border-line bg-surface px-6 py-3.5">
          <Textarea
            rows={2}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Send a message. Enter to send, Shift+Enter for a new line."
          />
          {streaming ? (
            <Button variant="danger" onClick={stop}>
              <Square /> Stop
            </Button>
          ) : (
            <Button variant="primary" onClick={send} disabled={!draft.trim()}>
              <Send /> Send
            </Button>
          )}
        </div>
      </div>

      <aside className="flex flex-col gap-4 overflow-auto border-line bg-ground p-5 lg:border-l">
        <div>
          <Eyebrow className="mb-2">Loaded</Eyebrow>
          <p className="text-[13px]">{selected?.name}</p>
          <p className="mt-0.5 font-mono text-[11px] text-faint">{selected?.base_model}</p>
        </div>

        <div>
          <Label htmlFor="sys">System prompt</Label>
          <Textarea
            id="sys"
            rows={3}
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="Optional. Sets how the model should behave."
          />
        </div>

        <Slider label="temperature" value={temperature} min={0} max={2} step={0.05} onChange={setTemperature} />
        <Slider label="top_p" value={topP} min={0.05} max={1} step={0.05} onChange={setTopP} />
        <Slider label="max_new_tokens" value={maxTokens} min={16} max={2048} step={16} onChange={setMaxTokens} />

        <div className="mt-auto flex flex-col gap-2">
          <Button size="sm" variant="ghost" onClick={() => setMessages([])}>
            Clear conversation
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={async () => {
              await api.closePlayground(sessionId).catch(() => {});
              setSessionId(null);
              setSelected(null);
              setMessages([]);
            }}
          >
            Unload model
          </Button>
        </div>
      </aside>
    </div>
  );
}

function Slider({
  label, value, min, max, step, onChange,
}: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <Label className="mb-0">{label}</Label>
        <span className="font-mono text-[11.5px] text-text tnum">{value}</span>
      </div>
      <input
        type="range"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[var(--brand)]"
      />
    </div>
  );
}
