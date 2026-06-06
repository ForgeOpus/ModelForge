import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, Boxes, Cpu, History, Play, Settings, Square, UploadCloud } from "lucide-react";
import "./styles.css";

const params = new URLSearchParams(window.location.search);
const urlToken = params.get("token");
if (urlToken) localStorage.setItem("modelforge_token", urlToken);
const token = localStorage.getItem("modelforge_token") || "";

async function api(path: string, init: RequestInit = {}) {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      "X-ModelForge-Token": token,
      ...(init.headers || {}),
    },
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function App() {
  const [view, setView] = useState("new");
  const [system, setSystem] = useState<any>(null);
  const [hardware, setHardware] = useState<any>(null);
  const [runs, setRuns] = useState<any[]>([]);
  const [events, setEvents] = useState<any[]>([]);
  const [form, setForm] = useState({
    use_case_slug: "instruction_following",
    objective: "supervised",
    model_id: "",
    dataset_path: "",
    dataset_remote: false,
    output_dir: "./modelforge_output",
  });
  const [recommendation, setRecommendation] = useState<any>(null);
  const [selectedRun, setSelectedRun] = useState<any>(null);
  const [error, setError] = useState("");

  const selectedUseCase = useMemo(
    () => system?.use_cases?.find((item: any) => item.slug === form.use_case_slug),
    [system, form.use_case_slug],
  );

  async function refresh() {
    setSystem(await api("/api/system"));
    setHardware(await api("/api/hardware"));
    setRuns((await api("/api/runs?limit=50")).runs || []);
  }

  useEffect(() => {
    refresh().catch((err) => setError(String(err)));
  }, []);

  async function getRecommendation() {
    setError("");
    const rec = await api("/api/recommendations", {
      method: "POST",
      body: JSON.stringify({
        use_case_slug: form.use_case_slug,
        objective: form.objective,
        n: 5,
      }),
    });
    setRecommendation(rec);
    setForm((prev) => ({ ...prev, model_id: rec.recommended_models?.[0]?.model_id || prev.model_id }));
  }

  async function startRun() {
    setError("");
    const run = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({
        ...form,
        recommendation,
      }),
    });
    setSelectedRun(run);
    setView("live");
    subscribe(run.id);
    await refresh();
  }

  function subscribe(runId: number) {
    const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${wsProtocol}://${window.location.host}/api/runs/${runId}/events?token=${token}`);
    socket.onmessage = (message) => {
      const event = JSON.parse(message.data);
      if (event.type !== "heartbeat") setEvents((prev) => [event, ...prev].slice(0, 200));
    };
  }

  return (
    <main className="shell">
      <aside>
        <div className="brand">ModelForge</div>
        <button className={view === "new" ? "active" : ""} onClick={() => setView("new")}><Play /> New Run</button>
        <button className={view === "live" ? "active" : ""} onClick={() => setView("live")}><Activity /> Live</button>
        <button className={view === "history" ? "active" : ""} onClick={() => setView("history")}><History /> History</button>
        <button className={view === "models" ? "active" : ""} onClick={() => setView("models")}><Boxes /> Artifacts</button>
        <button className={view === "settings" ? "active" : ""} onClick={() => setView("settings")}><Settings /> Settings</button>
      </aside>
      <section className="workspace">
        {error && <div className="error">{error}</div>}
        {view === "new" && (
          <div className="grid two">
            <section className="panel">
              <h1>New fine-tune</h1>
              <label>Use case</label>
              <select value={form.use_case_slug} onChange={(e) => setForm({ ...form, use_case_slug: e.target.value })}>
                {system?.use_cases?.map((item: any) => <option key={item.slug} value={item.slug}>{item.name}</option>)}
              </select>
              <p className="muted">{selectedUseCase?.description}</p>
              <label>Objective</label>
              <select value={form.objective} onChange={(e) => setForm({ ...form, objective: e.target.value })}>
                {(selectedUseCase?.compatible_objectives || ["supervised"]).map((item: string) => <option key={item} value={item}>{item}</option>)}
              </select>
              <button onClick={getRecommendation}>Generate recommendation</button>
              <label>Model</label>
              <input value={form.model_id} onChange={(e) => setForm({ ...form, model_id: e.target.value })} placeholder="org/model" />
              <label>Dataset</label>
              <input value={form.dataset_path} onChange={(e) => setForm({ ...form, dataset_path: e.target.value })} placeholder="path or HF dataset id" />
              <label className="inline"><input type="checkbox" checked={form.dataset_remote} onChange={(e) => setForm({ ...form, dataset_remote: e.target.checked })} /> HuggingFace dataset</label>
              <label>Output directory</label>
              <input value={form.output_dir} onChange={(e) => setForm({ ...form, output_dir: e.target.value })} />
              <button className="primary" disabled={!recommendation || !form.model_id || !form.dataset_path} onClick={startRun}>Start training</button>
            </section>
            <ConfigPanel title="Recommendation" data={recommendation} />
          </div>
        )}
        {view === "live" && (
          <section className="panel">
            <div className="toolbar">
              <h1>Live run</h1>
              {selectedRun && <button onClick={() => api(`/api/runs/${selectedRun.id}/cancel`, { method: "POST" })}><Square /> Cancel</button>}
            </div>
            <div className="log">
              {events.length === 0 && <p className="muted">No live events yet.</p>}
              {events.map((event, index) => <div key={index}><strong>{event.type}</strong> {event.message}</div>)}
            </div>
          </section>
        )}
        {view === "history" && (
          <section className="panel">
            <h1>Run history</h1>
            <table><tbody>{runs.map((run) => <tr key={run.id}><td>{run.id}</td><td>{run.status}</td><td>{run.model_id}</td><td>{run.output_dir}</td></tr>)}</tbody></table>
          </section>
        )}
        {view === "models" && (
          <section className="panel">
            <h1>Artifacts</h1>
            <p className="muted">Completed output directories appear in history. Hub push actions will attach here as model metadata matures.</p>
            <UploadCloud />
          </section>
        )}
        {view === "settings" && (
          <div className="grid two">
            <ConfigPanel title="System" data={system} />
            <ConfigPanel title="Hardware" data={hardware} icon={<Cpu />} />
          </div>
        )}
      </section>
    </main>
  );
}

function ConfigPanel({ title, data, icon }: { title: string; data: any; icon?: React.ReactNode }) {
  return (
    <section className="panel">
      <h1>{icon}{title}</h1>
      {!data ? <p className="muted">Nothing loaded.</p> : <pre>{JSON.stringify(data, null, 2)}</pre>}
    </section>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
